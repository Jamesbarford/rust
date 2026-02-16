use std::fmt::{self, Debug};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{ControlFlow, Deref};

use derive_where::derive_where;
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, HashStable_NoContext};
use rustc_type_ir_macros::{
    GenericTypeVisitable, Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic,
};
use tracing::instrument;

use crate::data_structures::SsoHashSet;
use crate::fold::{FallibleTypeFolder, TypeFoldable, TypeFolder, TypeSuperFoldable};
use crate::inherent::*;
use crate::lift::Lift;
use crate::visit::{
    Flags, TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor, VisitorResult,
};
use crate::{
    self as ty, CollectAndApply, DebruijnIndex, Interner, Mutability, TyKind, TypeFlags,
    UniverseIndex, WithCachedTypeInfo, try_visit,
};

/// `Binder` is a binder for higher-ranked lifetimes or types. It is part of the
/// compiler's representation for things like `for<'a> Fn(&'a isize)`
/// (which would be represented by the type `PolyTraitRef == Binder<I, TraitRef>`).
///
/// See <https://rustc-dev-guide.rust-lang.org/ty_module/instantiating_binders.html>
/// for more details.
///
/// `Decodable` and `Encodable` are implemented for `Binder<T>` using the `impl_binder_encode_decode!` macro.
// FIXME(derive-where#136): Need to use separate `derive_where` for
// `Copy` and `Ord` to prevent the emitted `Clone` and `PartialOrd`
// impls from incorrectly relying on `T: Copy` and `T: Ord`.
#[derive_where(Copy; I: Interner, T: Copy)]
#[derive_where(Clone, Hash, PartialEq, Debug; I: Interner, T)]
#[derive(GenericTypeVisitable)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub struct Binder<I: Interner, T> {
    value: T,
    bound_vars: I::BoundVarKinds,
}

impl<I: Interner, T: Eq> Eq for Binder<I, T> {}

// FIXME: We manually derive `Lift` because the `derive(Lift_Generic)` doesn't
// understand how to turn `T` to `T::Lifted` in the output `type Lifted`.
impl<I: Interner, U: Interner, T> Lift<U> for Binder<I, T>
where
    T: Lift<U>,
    I::BoundVarKinds: Lift<U, Lifted = U::BoundVarKinds>,
{
    type Lifted = Binder<U, T::Lifted>;

    fn lift_to_interner(self, cx: U) -> Option<Self::Lifted> {
        Some(Binder {
            value: self.value.lift_to_interner(cx)?,
            bound_vars: self.bound_vars.lift_to_interner(cx)?,
        })
    }
}

#[cfg(feature = "nightly")]
macro_rules! impl_binder_encode_decode {
    ($($t:ty),+ $(,)?) => {
        $(
            impl<I: Interner, E: rustc_serialize::Encoder> rustc_serialize::Encodable<E> for ty::Binder<I, $t>
            where
                $t: rustc_serialize::Encodable<E>,
                I::BoundVarKinds: rustc_serialize::Encodable<E>,
            {
                fn encode(&self, e: &mut E) {
                    self.bound_vars().encode(e);
                    self.as_ref().skip_binder().encode(e);
                }
            }
            impl<I: Interner, D: rustc_serialize::Decoder> rustc_serialize::Decodable<D> for ty::Binder<I, $t>
            where
                $t: TypeVisitable<I> + rustc_serialize::Decodable<D>,
                I::BoundVarKinds: rustc_serialize::Decodable<D>,
            {
                fn decode(decoder: &mut D) -> Self {
                    let bound_vars = rustc_serialize::Decodable::decode(decoder);
                    ty::Binder::bind_with_vars(rustc_serialize::Decodable::decode(decoder), bound_vars)
                }
            }
        )*
    }
}

#[cfg(feature = "nightly")]
impl_binder_encode_decode! {
    ty::FnSig<I>,
    ty::FnSigTys<I>,
    ty::TraitPredicate<I>,
    ty::ExistentialPredicate<I>,
    ty::TraitRef<I>,
    ty::ExistentialTraitRef<I>,
    ty::HostEffectPredicate<I>,
}

impl<I: Interner, T> Binder<I, T>
where
    T: TypeVisitable<I>,
{
    /// Wraps `value` in a binder, asserting that `value` does not
    /// contain any bound vars that would be bound by the
    /// binder. This is commonly used to 'inject' a value T into a
    /// different binding level.
    #[track_caller]
    pub fn dummy(value: T) -> Binder<I, T> {
        assert!(
            !value.has_escaping_bound_vars(),
            "`{value:?}` has escaping bound vars, so it cannot be wrapped in a dummy binder."
        );
        Binder { value, bound_vars: Default::default() }
    }

    pub fn bind_with_vars(value: T, bound_vars: I::BoundVarKinds) -> Binder<I, T> {
        if cfg!(debug_assertions) {
            let mut validator = ValidateBoundVars::new(bound_vars);
            let _ = value.visit_with(&mut validator);
        }
        Binder { value, bound_vars }
    }
}

impl<I: Interner, T: TypeFoldable<I>> TypeFoldable<I> for Binder<I, T> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_binder(self)
    }

    fn fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        folder.fold_binder(self)
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeVisitable<I> for Binder<I, T> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        visitor.visit_binder(self)
    }
}

impl<I: Interner, T: TypeFoldable<I>> TypeSuperFoldable<I> for Binder<I, T> {
    fn try_super_fold_with<F: FallibleTypeFolder<I>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        self.try_map_bound(|t| t.try_fold_with(folder))
    }

    fn super_fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        self.map_bound(|t| t.fold_with(folder))
    }
}

impl<I: Interner, T: TypeVisitable<I>> TypeSuperVisitable<I> for Binder<I, T> {
    fn super_visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        self.as_ref().skip_binder().visit_with(visitor)
    }
}

impl<I: Interner, T> Binder<I, T> {
    /// Returns the value contained inside of this `for<'a>`. Accessing generic args
    /// in the returned value is generally incorrect.
    ///
    /// Please read <https://rustc-dev-guide.rust-lang.org/ty_module/instantiating_binders.html>
    /// before using this function. It is usually better to discharge the binder using
    /// `no_bound_vars` or `instantiate_bound_regions` or something like that.
    ///
    /// `skip_binder` is only valid when you are either extracting data that does not reference
    /// any generic arguments, e.g. a `DefId`, or when you're making sure you only pass the
    /// value to things which can handle escaping bound vars.
    ///
    /// See existing uses of `.skip_binder()` in `rustc_trait_selection::traits::select`
    /// or `rustc_next_trait_solver` for examples.
    pub fn skip_binder(self) -> T {
        self.value
    }

    pub fn bound_vars(&self) -> I::BoundVarKinds {
        self.bound_vars
    }

    pub fn as_ref(&self) -> Binder<I, &T> {
        Binder { value: &self.value, bound_vars: self.bound_vars }
    }

    pub fn as_deref(&self) -> Binder<I, &T::Target>
    where
        T: Deref,
    {
        Binder { value: &self.value, bound_vars: self.bound_vars }
    }

    pub fn map_bound_ref<F, U: TypeVisitable<I>>(&self, f: F) -> Binder<I, U>
    where
        F: FnOnce(&T) -> U,
    {
        self.as_ref().map_bound(f)
    }

    pub fn map_bound<F, U: TypeVisitable<I>>(self, f: F) -> Binder<I, U>
    where
        F: FnOnce(T) -> U,
    {
        let Binder { value, bound_vars } = self;
        let value = f(value);
        if cfg!(debug_assertions) {
            let mut validator = ValidateBoundVars::new(bound_vars);
            let _ = value.visit_with(&mut validator);
        }
        Binder { value, bound_vars }
    }

    pub fn try_map_bound<F, U: TypeVisitable<I>, E>(self, f: F) -> Result<Binder<I, U>, E>
    where
        F: FnOnce(T) -> Result<U, E>,
    {
        let Binder { value, bound_vars } = self;
        let value = f(value)?;
        if cfg!(debug_assertions) {
            let mut validator = ValidateBoundVars::new(bound_vars);
            let _ = value.visit_with(&mut validator);
        }
        Ok(Binder { value, bound_vars })
    }

    /// Wraps a `value` in a binder, using the same bound variables as the
    /// current `Binder`. This should not be used if the new value *changes*
    /// the bound variables. Note: the (old or new) value itself does not
    /// necessarily need to *name* all the bound variables.
    ///
    /// This currently doesn't do anything different than `bind`, because we
    /// don't actually track bound vars. However, semantically, it is different
    /// because bound vars aren't allowed to change here, whereas they are
    /// in `bind`. This may be (debug) asserted in the future.
    pub fn rebind<U>(&self, value: U) -> Binder<I, U>
    where
        U: TypeVisitable<I>,
    {
        Binder::bind_with_vars(value, self.bound_vars)
    }

    /// Unwraps and returns the value within, but only if it contains
    /// no bound vars at all. (In other words, if this binder --
    /// and indeed any enclosing binder -- doesn't bind anything at
    /// all.) Otherwise, returns `None`.
    ///
    /// (One could imagine having a method that just unwraps a single
    /// binder, but permits late-bound vars bound by enclosing
    /// binders, but that would require adjusting the debruijn
    /// indices, and given the shallow binding structure we often use,
    /// would not be that useful.)
    pub fn no_bound_vars(self) -> Option<T>
    where
        T: TypeVisitable<I>,
    {
        // `self.value` is equivalent to `self.skip_binder()`
        if self.value.has_escaping_bound_vars() { None } else { Some(self.skip_binder()) }
    }
}

impl<I: Interner, T> Binder<I, Option<T>> {
    pub fn transpose(self) -> Option<Binder<I, T>> {
        let Binder { value, bound_vars } = self;
        value.map(|value| Binder { value, bound_vars })
    }
}

impl<I: Interner, T: IntoIterator> Binder<I, T> {
    pub fn iter(self) -> impl Iterator<Item = Binder<I, T::Item>> {
        let Binder { value, bound_vars } = self;
        value.into_iter().map(move |value| Binder { value, bound_vars })
    }
}

pub struct ValidateBoundVars<I: Interner> {
    bound_vars: I::BoundVarKinds,
    binder_index: ty::DebruijnIndex,
    // We only cache types because any complex const will have to step through
    // a type at some point anyways. We may encounter the same variable at
    // different levels of binding, so this can't just be `Ty`.
    visited: SsoHashSet<(ty::DebruijnIndex, Ty<I>)>,
}

impl<I: Interner> ValidateBoundVars<I> {
    pub fn new(bound_vars: I::BoundVarKinds) -> Self {
        ValidateBoundVars {
            bound_vars,
            binder_index: ty::INNERMOST,
            visited: SsoHashSet::default(),
        }
    }
}

impl<I: Interner> TypeVisitor<I> for ValidateBoundVars<I> {
    type Result = ControlFlow<()>;

    fn visit_binder<T: TypeVisitable<I>>(&mut self, t: &Binder<I, T>) -> Self::Result {
        self.binder_index.shift_in(1);
        let result = t.super_visit_with(self);
        self.binder_index.shift_out(1);
        result
    }

    fn visit_ty(&mut self, t: Ty<I>) -> Self::Result {
        if t.outer_exclusive_binder() < self.binder_index
            || !self.visited.insert((self.binder_index, t))
        {
            return ControlFlow::Break(());
        }
        match t.kind() {
            ty::Bound(ty::BoundVarIndexKind::Bound(debruijn), bound_ty)
                if debruijn == self.binder_index =>
            {
                let idx = bound_ty.var().as_usize();
                if self.bound_vars.len() <= idx {
                    panic!("Not enough bound vars: {:?} not found in {:?}", t, self.bound_vars);
                }
                bound_ty.assert_eq(self.bound_vars.get(idx).unwrap());
            }
            _ => {}
        };

        t.super_visit_with(self)
    }

    fn visit_const(&mut self, c: I::Const) -> Self::Result {
        if c.outer_exclusive_binder() < self.binder_index {
            return ControlFlow::Break(());
        }
        match c.kind() {
            ty::ConstKind::Bound(debruijn, bound_const)
                if debruijn == ty::BoundVarIndexKind::Bound(self.binder_index) =>
            {
                let idx = bound_const.var().as_usize();
                if self.bound_vars.len() <= idx {
                    panic!("Not enough bound vars: {:?} not found in {:?}", c, self.bound_vars);
                }
                bound_const.assert_eq(self.bound_vars.get(idx).unwrap());
            }
            _ => {}
        };

        c.super_visit_with(self)
    }

    fn visit_region(&mut self, r: I::Region) -> Self::Result {
        match r.kind() {
            ty::ReBound(index, br) if index == ty::BoundVarIndexKind::Bound(self.binder_index) => {
                let idx = br.var().as_usize();
                if self.bound_vars.len() <= idx {
                    panic!("Not enough bound vars: {:?} not found in {:?}", r, self.bound_vars);
                }
                br.assert_eq(self.bound_vars.get(idx).unwrap());
            }

            _ => (),
        };

        ControlFlow::Continue(())
    }
}

/// Similar to [`Binder`] except that it tracks early bound generics, i.e. `struct Foo<T>(T)`
/// needs `T` instantiated immediately. This type primarily exists to avoid forgetting to call
/// `instantiate`.
///
/// See <https://rustc-dev-guide.rust-lang.org/ty_module/early_binder.html> for more details.
// FIXME(derive-where#136): Need to use separate `derive_where` for
// `Copy` and `Ord` to prevent the emitted `Clone` and `PartialOrd`
// impls from incorrectly relying on `T: Copy` and `T: Ord`.
#[derive_where(Ord; I: Interner, T: Ord)]
#[derive_where(Copy; I: Interner, T: Copy)]
#[derive_where(Clone, PartialOrd, PartialEq, Hash, Debug; I: Interner, T)]
#[derive(GenericTypeVisitable)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub struct EarlyBinder<I: Interner, T> {
    value: T,
    #[derive_where(skip(Debug))]
    _tcx: PhantomData<fn() -> I>,
}

impl<I: Interner, T: Eq> Eq for EarlyBinder<I, T> {}

/// For early binders, you should first call `instantiate` before using any visitors.
#[cfg(feature = "nightly")]
impl<I: Interner, T> !TypeFoldable<I> for ty::EarlyBinder<I, T> {}

/// For early binders, you should first call `instantiate` before using any visitors.
#[cfg(feature = "nightly")]
impl<I: Interner, T> !TypeVisitable<I> for ty::EarlyBinder<I, T> {}

impl<I: Interner, T> EarlyBinder<I, T> {
    pub fn bind(value: T) -> EarlyBinder<I, T> {
        EarlyBinder { value, _tcx: PhantomData }
    }

    pub fn as_ref(&self) -> EarlyBinder<I, &T> {
        EarlyBinder { value: &self.value, _tcx: PhantomData }
    }

    pub fn map_bound_ref<F, U>(&self, f: F) -> EarlyBinder<I, U>
    where
        F: FnOnce(&T) -> U,
    {
        self.as_ref().map_bound(f)
    }

    pub fn map_bound<F, U>(self, f: F) -> EarlyBinder<I, U>
    where
        F: FnOnce(T) -> U,
    {
        let value = f(self.value);
        EarlyBinder { value, _tcx: PhantomData }
    }

    pub fn try_map_bound<F, U, E>(self, f: F) -> Result<EarlyBinder<I, U>, E>
    where
        F: FnOnce(T) -> Result<U, E>,
    {
        let value = f(self.value)?;
        Ok(EarlyBinder { value, _tcx: PhantomData })
    }

    pub fn rebind<U>(&self, value: U) -> EarlyBinder<I, U> {
        EarlyBinder { value, _tcx: PhantomData }
    }

    /// Skips the binder and returns the "bound" value. Accessing generic args
    /// in the returned value is generally incorrect.
    ///
    /// Please read <https://rustc-dev-guide.rust-lang.org/ty_module/early_binder.html>
    /// before using this function.
    ///
    /// Only use this to extract data that does not depend on generic parameters, e.g.
    /// to get the `DefId` of the inner value or the number of arguments ofan `FnSig`,
    /// or while making sure to only pass the value to functions which are explicitly
    /// set up to handle these uninstantiated generic parameters.
    ///
    /// To skip the binder on `x: &EarlyBinder<I, T>` to obtain `&T`, leverage
    /// [`EarlyBinder::as_ref`](EarlyBinder::as_ref): `x.as_ref().skip_binder()`.
    ///
    /// See also [`Binder::skip_binder`](Binder::skip_binder), which is
    /// the analogous operation on [`Binder`].
    pub fn skip_binder(self) -> T {
        self.value
    }
}

impl<I: Interner, T> EarlyBinder<I, Option<T>> {
    pub fn transpose(self) -> Option<EarlyBinder<I, T>> {
        self.value.map(|value| EarlyBinder { value, _tcx: PhantomData })
    }
}

impl<I: Interner, Iter: IntoIterator> EarlyBinder<I, Iter>
where
    Iter::Item: TypeFoldable<I>,
{
    pub fn iter_instantiated<A>(self, cx: I, args: A) -> IterInstantiated<I, Iter, A>
    where
        A: SliceLike<Item = I::GenericArg>,
    {
        IterInstantiated { it: self.value.into_iter(), cx, args }
    }

    /// Similar to [`instantiate_identity`](EarlyBinder::instantiate_identity),
    /// but on an iterator of `TypeFoldable` values.
    pub fn iter_identity(self) -> Iter::IntoIter {
        self.value.into_iter()
    }
}

pub struct IterInstantiated<I: Interner, Iter: IntoIterator, A> {
    it: Iter::IntoIter,
    cx: I,
    args: A,
}

impl<I: Interner, Iter: IntoIterator, A> Iterator for IterInstantiated<I, Iter, A>
where
    Iter::Item: TypeFoldable<I>,
    A: SliceLike<Item = I::GenericArg>,
{
    type Item = Iter::Item;

    fn next(&mut self) -> Option<Self::Item> {
        Some(
            EarlyBinder { value: self.it.next()?, _tcx: PhantomData }
                .instantiate(self.cx, self.args),
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<I: Interner, Iter: IntoIterator, A> DoubleEndedIterator for IterInstantiated<I, Iter, A>
where
    Iter::IntoIter: DoubleEndedIterator,
    Iter::Item: TypeFoldable<I>,
    A: SliceLike<Item = I::GenericArg>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(
            EarlyBinder { value: self.it.next_back()?, _tcx: PhantomData }
                .instantiate(self.cx, self.args),
        )
    }
}

impl<I: Interner, Iter: IntoIterator, A> ExactSizeIterator for IterInstantiated<I, Iter, A>
where
    Iter::IntoIter: ExactSizeIterator,
    Iter::Item: TypeFoldable<I>,
    A: SliceLike<Item = I::GenericArg>,
{
}

impl<'s, I: Interner, Iter: IntoIterator> EarlyBinder<I, Iter>
where
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy + TypeFoldable<I>,
{
    pub fn iter_instantiated_copied(
        self,
        cx: I,
        args: &'s [I::GenericArg],
    ) -> IterInstantiatedCopied<'s, I, Iter> {
        IterInstantiatedCopied { it: self.value.into_iter(), cx, args }
    }

    /// Similar to [`instantiate_identity`](EarlyBinder::instantiate_identity),
    /// but on an iterator of values that deref to a `TypeFoldable`.
    pub fn iter_identity_copied(self) -> IterIdentityCopied<Iter> {
        IterIdentityCopied { it: self.value.into_iter() }
    }
}

pub struct IterInstantiatedCopied<'a, I: Interner, Iter: IntoIterator> {
    it: Iter::IntoIter,
    cx: I,
    args: &'a [I::GenericArg],
}

impl<I: Interner, Iter: IntoIterator> Iterator for IterInstantiatedCopied<'_, I, Iter>
where
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy + TypeFoldable<I>,
{
    type Item = <Iter::Item as Deref>::Target;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|value| {
            EarlyBinder { value: *value, _tcx: PhantomData }.instantiate(self.cx, self.args)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<I: Interner, Iter: IntoIterator> DoubleEndedIterator for IterInstantiatedCopied<'_, I, Iter>
where
    Iter::IntoIter: DoubleEndedIterator,
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy + TypeFoldable<I>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.it.next_back().map(|value| {
            EarlyBinder { value: *value, _tcx: PhantomData }.instantiate(self.cx, self.args)
        })
    }
}

impl<I: Interner, Iter: IntoIterator> ExactSizeIterator for IterInstantiatedCopied<'_, I, Iter>
where
    Iter::IntoIter: ExactSizeIterator,
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy + TypeFoldable<I>,
{
}

pub struct IterIdentityCopied<Iter: IntoIterator> {
    it: Iter::IntoIter,
}

impl<Iter: IntoIterator> Iterator for IterIdentityCopied<Iter>
where
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy,
{
    type Item = <Iter::Item as Deref>::Target;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|i| *i)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<Iter: IntoIterator> DoubleEndedIterator for IterIdentityCopied<Iter>
where
    Iter::IntoIter: DoubleEndedIterator,
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.it.next_back().map(|i| *i)
    }
}

impl<Iter: IntoIterator> ExactSizeIterator for IterIdentityCopied<Iter>
where
    Iter::IntoIter: ExactSizeIterator,
    Iter::Item: Deref,
    <Iter::Item as Deref>::Target: Copy,
{
}
pub struct EarlyBinderIter<I, T> {
    t: T,
    _tcx: PhantomData<I>,
}

impl<I: Interner, T: IntoIterator> EarlyBinder<I, T> {
    pub fn transpose_iter(self) -> EarlyBinderIter<I, T::IntoIter> {
        EarlyBinderIter { t: self.value.into_iter(), _tcx: PhantomData }
    }
}

impl<I: Interner, T: Iterator> Iterator for EarlyBinderIter<I, T> {
    type Item = EarlyBinder<I, T::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        self.t.next().map(|value| EarlyBinder { value, _tcx: PhantomData })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.t.size_hint()
    }
}

impl<I: Interner, T: TypeFoldable<I>> ty::EarlyBinder<I, T> {
    pub fn instantiate<A>(self, cx: I, args: A) -> T
    where
        A: SliceLike<Item = I::GenericArg>,
    {
        // Nothing to fold, so let's avoid visiting things and possibly re-hashing/equating
        // them when interning. Perf testing found this to be a modest improvement.
        // See: <https://github.com/rust-lang/rust/pull/142317>
        if args.is_empty() {
            assert!(
                !self.value.has_param(),
                "{:?} has parameters, but no args were provided in instantiate",
                self.value,
            );
            return self.value;
        }
        let mut folder = ArgFolder { cx, args: args.as_slice(), binders_passed: 0 };
        self.value.fold_with(&mut folder)
    }

    /// Makes the identity replacement `T0 => T0, ..., TN => TN`.
    /// Conceptually, this converts universally bound variables into placeholders
    /// when inside of a given item.
    ///
    /// For example, consider `for<T> fn foo<T>(){ .. }`:
    /// - Outside of `foo`, `T` is bound (represented by the presence of `EarlyBinder`).
    /// - Inside of the body of `foo`, we treat `T` as a placeholder by calling
    /// `instantiate_identity` to discharge the `EarlyBinder`.
    pub fn instantiate_identity(self) -> T {
        self.value
    }

    /// Returns the inner value, but only if it contains no bound vars.
    pub fn no_bound_vars(self) -> Option<T> {
        if !self.value.has_param() { Some(self.value) } else { None }
    }
}

///////////////////////////////////////////////////////////////////////////
// The actual instantiation engine itself is a type folder.

struct ArgFolder<'a, I: Interner> {
    cx: I,
    args: &'a [I::GenericArg],

    /// Number of region binders we have passed through while doing the instantiation
    binders_passed: u32,
}

impl<'a, I: Interner> TypeFolder<I> for ArgFolder<'a, I> {
    #[inline]
    fn cx(&self) -> I {
        self.cx
    }

    fn fold_binder<T: TypeFoldable<I>>(&mut self, t: ty::Binder<I, T>) -> ty::Binder<I, T> {
        self.binders_passed += 1;
        let t = t.super_fold_with(self);
        self.binders_passed -= 1;
        t
    }

    fn fold_region(&mut self, r: I::Region) -> I::Region {
        // Note: This routine only handles regions that are bound on
        // type declarations and other outer declarations, not those
        // bound in *fn types*. Region instantiation of the bound
        // regions that appear in a function signature is done using
        // the specialized routine `ty::replace_late_regions()`.
        match r.kind() {
            ty::ReEarlyParam(data) => {
                let rk = self.args.get(data.index() as usize).map(|arg| arg.kind());
                match rk {
                    Some(ty::GenericArgKind::Lifetime(lt)) => self.shift_region_through_binders(lt),
                    Some(other) => self.region_param_expected(data, r, other),
                    None => self.region_param_out_of_range(data, r),
                }
            }
            ty::ReBound(..)
            | ty::ReLateParam(_)
            | ty::ReStatic
            | ty::RePlaceholder(_)
            | ty::ReErased
            | ty::ReError(_) => r,
            ty::ReVar(_) => panic!("unexpected region: {r:?}"),
        }
    }

    fn fold_ty(&mut self, t: Ty<I>) -> Ty<I> {
        if !t.has_param() {
            return t;
        }

        match t.kind() {
            ty::Param(p) => self.ty_for_param(p, t),
            _ => t.super_fold_with(self),
        }
    }

    fn fold_const(&mut self, c: I::Const) -> I::Const {
        if let ty::ConstKind::Param(p) = c.kind() {
            self.const_for_param(p, c)
        } else {
            c.super_fold_with(self)
        }
    }

    fn fold_predicate(&mut self, p: I::Predicate) -> I::Predicate {
        if p.has_param() { p.super_fold_with(self) } else { p }
    }

    fn fold_clauses(&mut self, c: I::Clauses) -> I::Clauses {
        if c.has_param() { c.super_fold_with(self) } else { c }
    }
}

impl<'a, I: Interner> ArgFolder<'a, I> {
    fn ty_for_param(&self, p: I::ParamTy, source_ty: Ty<I>) -> Ty<I> {
        // Look up the type in the args. It really should be in there.
        let opt_ty = self.args.get(p.index() as usize).map(|arg| arg.kind());
        let ty = match opt_ty {
            Some(ty::GenericArgKind::Type(ty)) => ty,
            Some(kind) => self.type_param_expected(p, source_ty, kind),
            None => self.type_param_out_of_range(p, source_ty),
        };

        self.shift_vars_through_binders(ty)
    }

    #[cold]
    #[inline(never)]
    fn type_param_expected(&self, p: I::ParamTy, ty: Ty<I>, kind: ty::GenericArgKind<I>) -> ! {
        panic!(
            "expected type for `{:?}` ({:?}/{}) but found {:?} when instantiating, args={:?}",
            p,
            ty,
            p.index(),
            kind,
            self.args,
        )
    }

    #[cold]
    #[inline(never)]
    fn type_param_out_of_range(&self, p: I::ParamTy, ty: Ty<I>) -> ! {
        panic!(
            "type parameter `{:?}` ({:?}/{}) out of range when instantiating, args={:?}",
            p,
            ty,
            p.index(),
            self.args,
        )
    }

    fn const_for_param(&self, p: I::ParamConst, source_ct: I::Const) -> I::Const {
        // Look up the const in the args. It really should be in there.
        let opt_ct = self.args.get(p.index() as usize).map(|arg| arg.kind());
        let ct = match opt_ct {
            Some(ty::GenericArgKind::Const(ct)) => ct,
            Some(kind) => self.const_param_expected(p, source_ct, kind),
            None => self.const_param_out_of_range(p, source_ct),
        };

        self.shift_vars_through_binders(ct)
    }

    #[cold]
    #[inline(never)]
    fn const_param_expected(
        &self,
        p: I::ParamConst,
        ct: I::Const,
        kind: ty::GenericArgKind<I>,
    ) -> ! {
        panic!(
            "expected const for `{:?}` ({:?}/{}) but found {:?} when instantiating args={:?}",
            p,
            ct,
            p.index(),
            kind,
            self.args,
        )
    }

    #[cold]
    #[inline(never)]
    fn const_param_out_of_range(&self, p: I::ParamConst, ct: I::Const) -> ! {
        panic!(
            "const parameter `{:?}` ({:?}/{}) out of range when instantiating args={:?}",
            p,
            ct,
            p.index(),
            self.args,
        )
    }

    #[cold]
    #[inline(never)]
    fn region_param_expected(
        &self,
        ebr: I::EarlyParamRegion,
        r: I::Region,
        kind: ty::GenericArgKind<I>,
    ) -> ! {
        panic!(
            "expected region for `{:?}` ({:?}/{}) but found {:?} when instantiating args={:?}",
            ebr,
            r,
            ebr.index(),
            kind,
            self.args,
        )
    }

    #[cold]
    #[inline(never)]
    fn region_param_out_of_range(&self, ebr: I::EarlyParamRegion, r: I::Region) -> ! {
        panic!(
            "region parameter `{:?}` ({:?}/{}) out of range when instantiating args={:?}",
            ebr,
            r,
            ebr.index(),
            self.args,
        )
    }

    /// It is sometimes necessary to adjust the De Bruijn indices during instantiation. This occurs
    /// when we are instantiating a type with escaping bound vars into a context where we have
    /// passed through binders. That's quite a mouthful. Let's see an example:
    ///
    /// ```
    /// type Func<A> = fn(A);
    /// type MetaFunc = for<'a> fn(Func<&'a i32>);
    /// ```
    ///
    /// The type `MetaFunc`, when fully expanded, will be
    /// ```ignore (illustrative)
    /// for<'a> fn(fn(&'a i32))
    /// //      ^~ ^~ ^~~
    /// //      |  |  |
    /// //      |  |  DebruijnIndex of 2
    /// //      Binders
    /// ```
    /// Here the `'a` lifetime is bound in the outer function, but appears as an argument of the
    /// inner one. Therefore, that appearance will have a DebruijnIndex of 2, because we must skip
    /// over the inner binder (remember that we count De Bruijn indices from 1). However, in the
    /// definition of `MetaFunc`, the binder is not visible, so the type `&'a i32` will have a
    /// De Bruijn index of 1. It's only during the instantiation that we can see we must increase the
    /// depth by 1 to account for the binder that we passed through.
    ///
    /// As a second example, consider this twist:
    ///
    /// ```
    /// type FuncTuple<A> = (A,fn(A));
    /// type MetaFuncTuple = for<'a> fn(FuncTuple<&'a i32>);
    /// ```
    ///
    /// Here the final type will be:
    /// ```ignore (illustrative)
    /// for<'a> fn((&'a i32, fn(&'a i32)))
    /// //          ^~~         ^~~
    /// //          |           |
    /// //   DebruijnIndex of 1 |
    /// //               DebruijnIndex of 2
    /// ```
    /// As indicated in the diagram, here the same type `&'a i32` is instantiated once, but in the
    /// first case we do not increase the De Bruijn index and in the second case we do. The reason
    /// is that only in the second case have we passed through a fn binder.
    #[instrument(level = "trace", skip(self), fields(binders_passed = self.binders_passed), ret)]
    fn shift_vars_through_binders<T: TypeFoldable<I>>(&self, val: T) -> T {
        if self.binders_passed == 0 || !val.has_escaping_bound_vars() {
            val
        } else {
            ty::shift_vars(self.cx, val, self.binders_passed)
        }
    }

    fn shift_region_through_binders(&self, region: I::Region) -> I::Region {
        if self.binders_passed == 0 || !region.has_escaping_bound_vars() {
            region
        } else {
            ty::shift_region(self.cx, region, self.binders_passed)
        }
    }
}

/// Okay, we do something fun for `Bound` types/regions/consts:
/// Specifically, we distinguish between *canonically* bound things and
/// `for<>` bound things. And, really, it comes down to caching during
/// canonicalization and instantiation.
///
/// To understand why we do this, imagine we have a type `(T, for<> fn(T))`.
/// If we just tracked canonically bound types with a `DebruijnIndex` (as we
/// used to), then the canonicalized type would be something like
/// `for<0> (^0.0, for<> fn(^1.0))` and so we can't cache `T -> ^0.0`,
/// we have to also factor in binder level. (Of course, we don't cache that
/// exactly, but rather the entire enclosing type, but the point stands.)
///
/// Of course, this is okay because we don't ever nest canonicalization, so
/// `BoundVarIndexKind::Canonical` is unambiguous. We, alternatively, could
/// have some sentinel `DebruijinIndex`, but that just seems too scary.
///
/// This doesn't seem to have a huge perf swing either way, but in the next
/// solver, canonicalization is hot and there are some pathological cases where
/// this is needed (`post-mono-higher-ranked-hang`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_NoContext)
)]
#[derive(TypeVisitable_Generic, GenericTypeVisitable, TypeFoldable_Generic)]
pub enum BoundVarIndexKind {
    Bound(DebruijnIndex),
    Canonical,
}

/// The "placeholder index" fully defines a placeholder region, type, or const. Placeholders are
/// identified by both a universe, as well as a name residing within that universe. Distinct bound
/// regions/types/consts within the same universe simply have an unknown relationship to one
// FIXME(derive-where#136): Need to use separate `derive_where` for
// `Copy` and `Ord` to prevent the emitted `Clone` and `PartialOrd`
// impls from incorrectly relying on `T: Copy` and `T: Ord`.
#[derive_where(Ord; I: Interner, T: Ord)]
#[derive_where(Copy; I: Interner, T: Copy)]
#[derive_where(Clone, PartialOrd, PartialEq, Eq, Hash; I: Interner, T)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub struct Placeholder<I: Interner, T> {
    pub universe: UniverseIndex,
    pub bound: T,
    #[type_foldable(identity)]
    #[type_visitable(ignore)]
    _tcx: PhantomData<fn() -> I>,
}

impl<I: Interner, T: fmt::Debug> fmt::Debug for ty::Placeholder<I, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.universe == ty::UniverseIndex::ROOT {
            write!(f, "!{:?}", self.bound)
        } else {
            write!(f, "!{}_{:?}", self.universe.index(), self.bound)
        }
    }
}

impl<I: Interner, U: Interner, T> Lift<U> for Placeholder<I, T>
where
    T: Lift<U>,
{
    type Lifted = Placeholder<U, T::Lifted>;

    fn lift_to_interner(self, cx: U) -> Option<Self::Lifted> {
        Some(Placeholder {
            universe: self.universe,
            bound: self.bound.lift_to_interner(cx)?,
            _tcx: PhantomData,
        })
    }
}

#[derive_where(Clone, Copy, PartialEq, Eq, Hash; I: Interner)]
#[derive(Lift_Generic)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]

pub enum BoundRegionKind<I: Interner> {
    /// An anonymous region parameter for a given fn (&T)
    Anon,

    /// An anonymous region parameter with a `Symbol` name.
    ///
    /// Used to give late-bound regions names for things like pretty printing.
    NamedForPrinting(I::Symbol),

    /// Late-bound regions that appear in the AST.
    Named(I::DefId),

    /// Anonymous region for the implicit env pointer parameter
    /// to a closure
    ClosureEnv,
}

impl<I: Interner> fmt::Debug for ty::BoundRegionKind<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ty::BoundRegionKind::Anon => write!(f, "BrAnon"),
            ty::BoundRegionKind::NamedForPrinting(name) => {
                write!(f, "BrNamedForPrinting({:?})", name)
            }
            ty::BoundRegionKind::Named(did) => {
                write!(f, "BrNamed({did:?})")
            }
            ty::BoundRegionKind::ClosureEnv => write!(f, "BrEnv"),
        }
    }
}

impl<I: Interner> BoundRegionKind<I> {
    pub fn is_named(&self, tcx: I) -> bool {
        self.get_name(tcx).is_some()
    }

    pub fn get_name(&self, tcx: I) -> Option<I::Symbol> {
        match *self {
            ty::BoundRegionKind::Named(def_id) => {
                let name = tcx.item_name(def_id);
                if name.is_kw_underscore_lifetime() { None } else { Some(name) }
            }
            ty::BoundRegionKind::NamedForPrinting(name) => Some(name),
            _ => None,
        }
    }

    pub fn get_id(&self) -> Option<I::DefId> {
        match *self {
            ty::BoundRegionKind::Named(id) => Some(id),
            _ => None,
        }
    }
}

#[derive_where(Clone, Copy, PartialEq, Eq, Debug, Hash; I: Interner)]
#[derive(Lift_Generic)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
pub enum BoundTyKind<I: Interner> {
    Anon,
    Param(I::DefId),
}

#[derive_where(Clone, Copy, PartialEq, Eq, Debug, Hash; I: Interner)]
#[derive(Lift_Generic)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_NoContext)
)]
pub enum BoundVariableKind<I: Interner> {
    Ty(BoundTyKind<I>),
    Region(BoundRegionKind<I>),
    Const,
}

impl<I: Interner> BoundVariableKind<I> {
    pub fn expect_region(self) -> BoundRegionKind<I> {
        match self {
            BoundVariableKind::Region(lt) => lt,
            _ => panic!("expected a region, but found another kind"),
        }
    }

    pub fn expect_ty(self) -> BoundTyKind<I> {
        match self {
            BoundVariableKind::Ty(ty) => ty,
            _ => panic!("expected a type, but found another kind"),
        }
    }

    pub fn expect_const(self) {
        match self {
            BoundVariableKind::Const => (),
            _ => panic!("expected a const, but found another kind"),
        }
    }
}

#[derive_where(Clone, Copy, PartialEq, Eq, Hash; I: Interner)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, HashStable_NoContext, Decodable_NoContext)
)]
pub struct BoundRegion<I: Interner> {
    pub var: ty::BoundVar,
    pub kind: BoundRegionKind<I>,
}

impl<I: Interner> core::fmt::Debug for BoundRegion<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            BoundRegionKind::Anon => write!(f, "{:?}", self.var),
            BoundRegionKind::ClosureEnv => write!(f, "{:?}.Env", self.var),
            BoundRegionKind::Named(def) => {
                write!(f, "{:?}.Named({:?})", self.var, def)
            }
            BoundRegionKind::NamedForPrinting(symbol) => {
                write!(f, "{:?}.NamedAnon({:?})", self.var, symbol)
            }
        }
    }
}

impl<I: Interner> BoundRegion<I> {
    pub fn var(self) -> ty::BoundVar {
        self.var
    }

    pub fn assert_eq(self, var: BoundVariableKind<I>) {
        assert_eq!(self.kind, var.expect_region())
    }
}

pub type PlaceholderRegion<I> = ty::Placeholder<I, BoundRegion<I>>;

impl<I: Interner> PlaceholderRegion<I> {
    pub fn universe(self) -> UniverseIndex {
        self.universe
    }

    pub fn var(self) -> ty::BoundVar {
        self.bound.var()
    }

    pub fn with_updated_universe(self, ui: UniverseIndex) -> Self {
        Self { universe: ui, bound: self.bound, _tcx: PhantomData }
    }

    pub fn new(ui: UniverseIndex, bound: BoundRegion<I>) -> Self {
        Self { universe: ui, bound, _tcx: PhantomData }
    }

    pub fn new_anon(ui: UniverseIndex, var: ty::BoundVar) -> Self {
        let bound = BoundRegion { var, kind: BoundRegionKind::Anon };
        Self { universe: ui, bound, _tcx: PhantomData }
    }
}

#[derive_where(Clone, Copy, PartialEq, Eq, Hash; I: Interner)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_NoContext)
)]
pub struct BoundTy<I: Interner> {
    pub var: ty::BoundVar,
    pub kind: BoundTyKind<I>,
}

impl<I: Interner, U: Interner> Lift<U> for BoundTy<I>
where
    BoundTyKind<I>: Lift<U, Lifted = BoundTyKind<U>>,
{
    type Lifted = BoundTy<U>;

    fn lift_to_interner(self, cx: U) -> Option<Self::Lifted> {
        Some(BoundTy { var: self.var, kind: self.kind.lift_to_interner(cx)? })
    }
}

impl<I: Interner> fmt::Debug for ty::BoundTy<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            ty::BoundTyKind::Anon => write!(f, "{:?}", self.var),
            ty::BoundTyKind::Param(def_id) => write!(f, "{def_id:?}"),
        }
    }
}

impl<I: Interner> BoundTy<I> {
    pub fn var(self) -> ty::BoundVar {
        self.var
    }

    pub fn assert_eq(self, var: BoundVariableKind<I>) {
        assert_eq!(self.kind, var.expect_ty())
    }
}

pub type PlaceholderType<I> = ty::Placeholder<I, BoundTy<I>>;

impl<I: Interner> PlaceholderType<I> {
    pub fn universe(self) -> UniverseIndex {
        self.universe
    }

    pub fn var(self) -> ty::BoundVar {
        self.bound.var
    }

    pub fn with_updated_universe(self, ui: UniverseIndex) -> Self {
        Self { universe: ui, bound: self.bound, _tcx: PhantomData }
    }

    pub fn new(ui: UniverseIndex, bound: BoundTy<I>) -> Self {
        Self { universe: ui, bound, _tcx: PhantomData }
    }

    pub fn new_anon(ui: UniverseIndex, var: ty::BoundVar) -> Self {
        let bound = BoundTy { var, kind: BoundTyKind::Anon };
        Self { universe: ui, bound, _tcx: PhantomData }
    }
}

#[derive_where(Clone, Copy, PartialEq, Debug, Eq, Hash; I: Interner)]
#[cfg_attr(
    feature = "nightly",
    derive(Encodable_NoContext, Decodable_NoContext, HashStable_NoContext)
)]
pub struct BoundConst<I: Interner> {
    pub var: ty::BoundVar,
    #[derive_where(skip(Debug))]
    pub _tcx: PhantomData<fn() -> I>,
}

impl<I: Interner> BoundConst<I> {
    pub fn var(self) -> ty::BoundVar {
        self.var
    }

    pub fn assert_eq(self, var: BoundVariableKind<I>) {
        var.expect_const()
    }

    pub fn new(var: ty::BoundVar) -> Self {
        Self { var, _tcx: PhantomData }
    }
}

pub type PlaceholderConst<I> = ty::Placeholder<I, BoundConst<I>>;

impl<I: Interner> PlaceholderConst<I> {
    pub fn universe(self) -> UniverseIndex {
        self.universe
    }

    pub fn var(self) -> ty::BoundVar {
        self.bound.var
    }

    pub fn with_updated_universe(self, ui: UniverseIndex) -> Self {
        Self { universe: ui, bound: self.bound, _tcx: PhantomData }
    }

    pub fn new(ui: UniverseIndex, bound: BoundConst<I>) -> Self {
        Self { universe: ui, bound, _tcx: PhantomData }
    }

    pub fn new_anon(ui: UniverseIndex, var: ty::BoundVar) -> Self {
        let bound = BoundConst::new(var);
        Self { universe: ui, bound, _tcx: PhantomData }
    }

    pub fn find_const_ty_from_env(self, env: I::ParamEnv) -> Ty<I> {
        let mut candidates = env.caller_bounds().iter().filter_map(|clause| {
            // `ConstArgHasType` are never desugared to be higher ranked.
            match clause.kind().skip_binder() {
                ty::ClauseKind::ConstArgHasType(placeholder_ct, ty) => {
                    assert!(!(placeholder_ct, ty).has_escaping_bound_vars());

                    match placeholder_ct.kind() {
                        ty::ConstKind::Placeholder(placeholder_ct) if placeholder_ct == self => {
                            Some(ty)
                        }
                        _ => None,
                    }
                }
                _ => None,
            }
        });

        // N.B. it may be tempting to fix ICEs by making this function return
        // `Option<Ty<'tcx>>` instead of `Ty<'tcx>`; however, this is generally
        // considered to be a bandaid solution, since it hides more important
        // underlying issues with how we construct generics and predicates of
        // items. It's advised to fix the underlying issue rather than trying
        // to modify this function.
        let ty = candidates.next().unwrap_or_else(|| {
            panic!("cannot find `{self:?}` in param-env: {env:#?}");
        });
        assert!(
            candidates.next().is_none(),
            "did not expect duplicate `ConstParamHasTy` for `{self:?}` in param-env: {env:#?}"
        );
        ty
    }
}

/// Use this rather than `TyKind`, whenever possible.
#[derive_where(Copy; I: Interner, I::Interned<WithCachedTypeInfo<TyKind<I>>>: Copy)]
#[derive_where(Clone, PartialEq, Eq, Hash; I: Interner)]
#[cfg_attr(feature = "nightly", derive(HashStable_NoContext))]
#[rustc_diagnostic_item = "Ty"]
#[rustc_pass_by_value]
#[rustc_has_incoherent_inherent_impls]
pub struct Ty<I: Interner>(pub I::Interned<WithCachedTypeInfo<TyKind<I>>>);

impl<I: Interner> Ty<I> {
    #[inline]
    pub fn from_interned(interned: I::Interned<WithCachedTypeInfo<TyKind<I>>>) -> Self {
        Ty(interned)
    }

    #[inline]
    pub fn interned(self) -> I::Interned<WithCachedTypeInfo<TyKind<I>>> {
        self.0
    }

    #[inline]
    #[allow(rustc::pass_by_value)]
    pub fn with_cached_type_info(&self) -> &WithCachedTypeInfo<TyKind<I>>
    where
        I::Interned<WithCachedTypeInfo<TyKind<I>>>: Deref<Target = WithCachedTypeInfo<TyKind<I>>>,
    {
        &*self.0
    }
}

impl<I: Interner> fmt::Debug for Ty<I>
where
    I::Interned<WithCachedTypeInfo<TyKind<I>>>: Deref<Target = WithCachedTypeInfo<TyKind<I>>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&(*self).kind(), f)
    }
}

impl<I: Interner> IntoKind for Ty<I>
where
    I::Interned<WithCachedTypeInfo<TyKind<I>>>: Deref<Target = WithCachedTypeInfo<TyKind<I>>>,
{
    type Kind = TyKind<I>;

    #[inline]
    fn kind(self) -> TyKind<I> {
        (*self.0).internee
    }
}

impl<I: Interner> Flags for Ty<I>
where
    I::Interned<WithCachedTypeInfo<TyKind<I>>>: Deref<Target = WithCachedTypeInfo<TyKind<I>>>,
{
    #[inline]
    fn flags(&self) -> TypeFlags {
        self.0.flags
    }

    #[inline]
    fn outer_exclusive_binder(&self) -> DebruijnIndex {
        self.0.outer_exclusive_binder
    }
}

impl<I: Interner> TypeVisitable<I> for Ty<I> {
    fn visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        visitor.visit_ty(*self)
    }
}

impl<I: Interner> TypeSuperVisitable<I> for Ty<I>
where
    I::Interned<WithCachedTypeInfo<TyKind<I>>>: Deref<Target = WithCachedTypeInfo<TyKind<I>>>,
    I::BoundExistentialPredicates: TypeVisitable<I>,
    I::Const: TypeVisitable<I>,
    I::ErrorGuaranteed: TypeVisitable<I>,
    I::GenericArgs: TypeVisitable<I>,
    I::Pat: TypeVisitable<I>,
    I::Region: TypeVisitable<I>,
    I::Tys: TypeVisitable<I>,
{
    fn super_visit_with<V: TypeVisitor<I>>(&self, visitor: &mut V) -> V::Result {
        match (*self).kind() {
            ty::RawPtr(ty, _mutbl) => ty.visit_with(visitor),
            ty::Array(typ, sz) => {
                try_visit!(typ.visit_with(visitor));
                sz.visit_with(visitor)
            }
            ty::Slice(typ) => typ.visit_with(visitor),
            ty::Adt(_, args) => args.visit_with(visitor),
            ty::Dynamic(trait_ty, reg) => {
                try_visit!(trait_ty.visit_with(visitor));
                reg.visit_with(visitor)
            }
            ty::Tuple(ts) => ts.visit_with(visitor),
            ty::FnDef(_, args) => args.visit_with(visitor),
            ty::FnPtr(sig_tys, _) => sig_tys.visit_with(visitor),
            ty::UnsafeBinder(f) => f.visit_with(visitor),
            ty::Ref(r, ty, _) => {
                try_visit!(r.visit_with(visitor));
                ty.visit_with(visitor)
            }
            ty::Coroutine(_did, args) => args.visit_with(visitor),
            ty::CoroutineWitness(_did, args) => args.visit_with(visitor),
            ty::Closure(_did, args) => args.visit_with(visitor),
            ty::CoroutineClosure(_did, args) => args.visit_with(visitor),
            ty::Alias(_, data) => data.visit_with(visitor),
            ty::Pat(ty, pat) => {
                try_visit!(ty.visit_with(visitor));
                pat.visit_with(visitor)
            }
            ty::Error(guar) => guar.visit_with(visitor),

            ty::Bool
            | ty::Char
            | ty::Str
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Infer(_)
            | ty::Param(..)
            | ty::Bound(..)
            | ty::Placeholder(..)
            | ty::Never
            | ty::Foreign(..) => V::Result::output(),
        }
    }
}

impl<I: Interner> TypeFoldable<I> for Ty<I> {
    fn try_fold_with<F: FallibleTypeFolder<I>>(self, folder: &mut F) -> Result<Self, F::Error> {
        folder.try_fold_ty(self)
    }

    fn fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        folder.fold_ty(self)
    }
}

impl<I: Interner> TypeSuperFoldable<I> for Ty<I>
where
    I::Interned<WithCachedTypeInfo<TyKind<I>>>: Deref<Target = WithCachedTypeInfo<TyKind<I>>>,
    I::BoundExistentialPredicates: TypeFoldable<I>,
    I::Const: TypeFoldable<I>,
    I::GenericArgs: TypeFoldable<I>,
    I::Pat: TypeFoldable<I>,
    I::Region: TypeFoldable<I>,
    I::Tys: TypeFoldable<I>,
{
    fn try_super_fold_with<F: FallibleTypeFolder<I>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        let kind = match self.kind() {
            ty::RawPtr(ty, mutbl) => ty::RawPtr(ty.try_fold_with(folder)?, mutbl),
            ty::Array(typ, sz) => ty::Array(typ.try_fold_with(folder)?, sz.try_fold_with(folder)?),
            ty::Slice(typ) => ty::Slice(typ.try_fold_with(folder)?),
            ty::Adt(tid, args) => ty::Adt(tid, args.try_fold_with(folder)?),
            ty::Dynamic(trait_ty, region) => {
                ty::Dynamic(trait_ty.try_fold_with(folder)?, region.try_fold_with(folder)?)
            }
            ty::Tuple(ts) => ty::Tuple(ts.try_fold_with(folder)?),
            ty::FnDef(def_id, args) => ty::FnDef(def_id, args.try_fold_with(folder)?),
            ty::FnPtr(sig_tys, hdr) => ty::FnPtr(sig_tys.try_fold_with(folder)?, hdr),
            ty::UnsafeBinder(f) => ty::UnsafeBinder(f.try_fold_with(folder)?),
            ty::Ref(r, ty, mutbl) => {
                ty::Ref(r.try_fold_with(folder)?, ty.try_fold_with(folder)?, mutbl)
            }
            ty::Coroutine(did, args) => ty::Coroutine(did, args.try_fold_with(folder)?),
            ty::CoroutineWitness(did, args) => {
                ty::CoroutineWitness(did, args.try_fold_with(folder)?)
            }
            ty::Closure(did, args) => ty::Closure(did, args.try_fold_with(folder)?),
            ty::CoroutineClosure(did, args) => {
                ty::CoroutineClosure(did, args.try_fold_with(folder)?)
            }
            ty::Alias(kind, data) => ty::Alias(kind, data.try_fold_with(folder)?),
            ty::Pat(ty, pat) => ty::Pat(ty.try_fold_with(folder)?, pat.try_fold_with(folder)?),

            ty::Bool
            | ty::Char
            | ty::Str
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Error(_)
            | ty::Infer(_)
            | ty::Param(..)
            | ty::Bound(..)
            | ty::Placeholder(..)
            | ty::Never
            | ty::Foreign(..) => return Ok(self),
        };

        Ok(if self.kind() == kind { self } else { folder.cx().mk_ty_from_kind(kind) })
    }

    fn super_fold_with<F: TypeFolder<I>>(self, folder: &mut F) -> Self {
        let kind = match self.kind() {
            ty::RawPtr(ty, mutbl) => ty::RawPtr(ty.fold_with(folder), mutbl),
            ty::Array(typ, sz) => ty::Array(typ.fold_with(folder), sz.fold_with(folder)),
            ty::Slice(typ) => ty::Slice(typ.fold_with(folder)),
            ty::Adt(tid, args) => ty::Adt(tid, args.fold_with(folder)),
            ty::Dynamic(trait_ty, region) => {
                ty::Dynamic(trait_ty.fold_with(folder), region.fold_with(folder))
            }
            ty::Tuple(ts) => ty::Tuple(ts.fold_with(folder)),
            ty::FnDef(def_id, args) => ty::FnDef(def_id, args.fold_with(folder)),
            ty::FnPtr(sig_tys, hdr) => ty::FnPtr(sig_tys.fold_with(folder), hdr),
            ty::UnsafeBinder(f) => ty::UnsafeBinder(f.fold_with(folder)),
            ty::Ref(r, ty, mutbl) => ty::Ref(r.fold_with(folder), ty.fold_with(folder), mutbl),
            ty::Coroutine(did, args) => ty::Coroutine(did, args.fold_with(folder)),
            ty::CoroutineWitness(did, args) => ty::CoroutineWitness(did, args.fold_with(folder)),
            ty::Closure(did, args) => ty::Closure(did, args.fold_with(folder)),
            ty::CoroutineClosure(did, args) => ty::CoroutineClosure(did, args.fold_with(folder)),
            ty::Alias(kind, data) => ty::Alias(kind, data.fold_with(folder)),
            ty::Pat(ty, pat) => ty::Pat(ty.fold_with(folder), pat.fold_with(folder)),

            ty::Bool
            | ty::Char
            | ty::Str
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Error(_)
            | ty::Infer(_)
            | ty::Param(..)
            | ty::Bound(..)
            | ty::Placeholder(..)
            | ty::Never
            | ty::Foreign(..) => return self,
        };

        if self.kind() == kind { self } else { folder.cx().mk_ty_from_kind(kind) }
    }
}

impl<I: Interner> Ty<I>
where
    I::BoundExistentialPredicates: TypeFoldable<I> + TypeVisitable<I>,
    I::GenericArg: From<Ty<I>>,
    I::GenericArgs: TypeFoldable<I> + TypeVisitable<I>,
    I::Const: TypeFoldable<I> + TypeVisitable<I>,
    I::ErrorGuaranteed: TypeVisitable<I>,
    I::Pat: TypeFoldable<I> + TypeVisitable<I>,
    I::Region: TypeFoldable<I> + TypeVisitable<I>,
    I::Term: From<Ty<I>>,
    I::Tys: TypeFoldable<I> + TypeVisitable<I>,
    I::Interned<WithCachedTypeInfo<TyKind<I>>>:
        Copy + Clone + Debug + Hash + Eq + Deref<Target = WithCachedTypeInfo<TyKind<I>>>,
{
    /// Avoid using this in favour of more specific `new_*` methods, where possible.
    /// The more specific methods will often optimize their creation.
    #[allow(rustc::usage_of_ty_tykind)]
    #[inline]
    pub fn new(interner: I, st: TyKind<I>) -> Self {
        interner.mk_ty_from_kind(st)
    }

    pub fn new_unit(interner: I) -> Self {
        Self::new(interner, ty::Tuple(Default::default()))
    }

    pub fn new_bool(interner: I) -> Self {
        Self::new(interner, ty::Bool)
    }

    pub fn new_u8(interner: I) -> Self {
        Self::new(interner, ty::Uint(ty::UintTy::U8))
    }

    pub fn new_usize(interner: I) -> Self {
        Self::new(interner, ty::Uint(ty::UintTy::Usize))
    }

    pub fn new_infer(interner: I, var: ty::InferTy) -> Self {
        Self::new(interner, ty::Infer(var))
    }

    pub fn new_var(interner: I, var: ty::TyVid) -> Self {
        Self::new(interner, ty::Infer(ty::InferTy::TyVar(var)))
    }

    pub fn new_param(interner: I, param: I::ParamTy) -> Self {
        Self::new(interner, ty::Param(param))
    }

    pub fn new_placeholder(interner: I, param: ty::PlaceholderType<I>) -> Self {
        Self::new(interner, ty::Placeholder(param))
    }

    pub fn new_bound(interner: I, debruijn: ty::DebruijnIndex, var: ty::BoundTy<I>) -> Self {
        Self::new(interner, ty::Bound(ty::BoundVarIndexKind::Bound(debruijn), var))
    }

    pub fn new_anon_bound(interner: I, debruijn: ty::DebruijnIndex, var: ty::BoundVar) -> Self {
        let bound_ty = ty::BoundTy { var, kind: ty::BoundTyKind::Anon };
        Self::new(interner, ty::Bound(ty::BoundVarIndexKind::Bound(debruijn), bound_ty))
    }

    pub fn new_canonical_bound(interner: I, var: ty::BoundVar) -> Self {
        let bound_ty = ty::BoundTy { var, kind: ty::BoundTyKind::Anon };
        Self::new(interner, ty::Bound(ty::BoundVarIndexKind::Canonical, bound_ty))
    }

    pub fn new_alias(interner: I, kind: ty::AliasTyKind, alias_ty: ty::AliasTy<I>) -> Self {
        Self::new(interner, ty::Alias(kind, alias_ty))
    }

    pub fn new_projection_from_args(interner: I, def_id: I::DefId, args: I::GenericArgs) -> Self {
        Ty::new_alias(
            interner,
            ty::AliasTyKind::Projection,
            ty::AliasTy::new_from_args(interner, def_id, args),
        )
    }

    pub fn new_projection(
        interner: I,
        def_id: I::DefId,
        args: impl IntoIterator<Item: Into<I::GenericArg>>,
    ) -> Self {
        Ty::new_alias(
            interner,
            ty::AliasTyKind::Projection,
            ty::AliasTy::new(interner, def_id, args),
        )
    }

    /// Constructs a `TyKind::Error` type with current `ErrorGuaranteed`
    pub fn new_error(interner: I, guar: I::ErrorGuaranteed) -> Self {
        Self::new(interner, ty::Error(guar))
    }

    pub fn new_adt(interner: I, adt_def: I::AdtDef, args: I::GenericArgs) -> Self {
        Self::new(interner, ty::Adt(adt_def, args))
    }

    pub fn new_foreign(interner: I, def_id: I::ForeignId) -> Self {
        Self::new(interner, ty::Foreign(def_id))
    }

    pub fn new_dynamic(
        interner: I,
        preds: I::BoundExistentialPredicates,
        region: I::Region,
    ) -> Self {
        Self::new(interner, ty::Dynamic(preds, region))
    }

    pub fn new_coroutine(interner: I, def_id: I::CoroutineId, args: I::GenericArgs) -> Self {
        Self::new(interner, ty::Coroutine(def_id, args))
    }

    pub fn new_coroutine_closure(
        interner: I,
        def_id: I::CoroutineClosureId,
        args: I::GenericArgs,
    ) -> Self {
        Self::new(interner, ty::CoroutineClosure(def_id, args))
    }

    pub fn new_closure(interner: I, def_id: I::ClosureId, args: I::GenericArgs) -> Self {
        Self::new(interner, ty::Closure(def_id, args))
    }

    pub fn new_coroutine_witness(
        interner: I,
        def_id: I::CoroutineId,
        args: I::GenericArgs,
    ) -> Self {
        Self::new(interner, ty::CoroutineWitness(def_id, args))
    }

    pub fn new_coroutine_witness_for_coroutine(
        interner: I,
        def_id: I::CoroutineId,
        coroutine_args: I::GenericArgs,
    ) -> Self {
        interner.mk_coroutine_witness_for_coroutine(def_id, coroutine_args)
    }

    pub fn new_ptr(interner: I, ty: Self, mutbl: Mutability) -> Self {
        Self::new(interner, ty::RawPtr(ty, mutbl))
    }

    pub fn new_ref(interner: I, region: I::Region, ty: Self, mutbl: Mutability) -> Self {
        Self::new(interner, ty::Ref(region, ty, mutbl))
    }

    pub fn new_array_with_const_len(interner: I, ty: Self, len: I::Const) -> Self {
        Self::new(interner, ty::Array(ty, len))
    }

    pub fn new_slice(interner: I, ty: Self) -> Self {
        Self::new(interner, ty::Slice(ty))
    }

    pub fn new_tup(interner: I, tys: &[Ty<I>]) -> Self {
        if tys.is_empty() {
            return Self::new_unit(interner);
        }

        let tys = interner.mk_type_list_from_iter(tys.iter().copied());
        Self::new(interner, ty::Tuple(tys))
    }

    pub fn new_tup_from_iter<It, T>(interner: I, iter: It) -> T::Output
    where
        It: Iterator<Item = T>,
        T: CollectAndApply<Self, Self>,
    {
        T::collect_and_apply(iter, |ts| Self::new_tup(interner, ts))
    }

    pub fn new_fn_def(interner: I, def_id: I::FunctionId, args: I::GenericArgs) -> Self {
        Self::new(interner, ty::FnDef(def_id, args))
    }

    pub fn new_fn_ptr(interner: I, sig: ty::Binder<I, ty::FnSig<I>>) -> Self {
        let (sig_tys, hdr) = sig.split();
        Self::new(interner, ty::FnPtr(sig_tys, hdr))
    }

    pub fn new_pat(interner: I, ty: Self, pat: I::Pat) -> Self {
        Self::new(interner, ty::Pat(ty, pat))
    }

    pub fn new_unsafe_binder(interner: I, ty: ty::Binder<I, Ty<I>>) -> Self {
        Self::new(interner, ty::UnsafeBinder(ty::UnsafeBinderInner::from(ty)))
    }

    pub fn tuple_fields(self) -> I::Tys {
        match self.kind() {
            ty::Tuple(tys) => tys,
            _ => panic!("tuple_fields called on non-tuple: {self:?}"),
        }
    }

    pub fn to_opt_closure_kind(self) -> Option<ty::ClosureKind> {
        match self.kind() {
            ty::Int(int_ty) => match int_ty {
                ty::IntTy::I8 => Some(ty::ClosureKind::Fn),
                ty::IntTy::I16 => Some(ty::ClosureKind::FnMut),
                ty::IntTy::I32 => Some(ty::ClosureKind::FnOnce),
                _ => panic!("cannot convert type `{self:?}` to a closure kind"),
            },

            ty::Bound(..) | ty::Placeholder(_) | ty::Param(_) | ty::Infer(_) => None,

            ty::Error(_) => Some(ty::ClosureKind::Fn),

            _ => panic!("cannot convert type `{self:?}` to a closure kind"),
        }
    }

    pub fn from_closure_kind(interner: I, kind: ty::ClosureKind) -> Self {
        let int_ty = match kind {
            ty::ClosureKind::Fn => ty::IntTy::I8,
            ty::ClosureKind::FnMut => ty::IntTy::I16,
            ty::ClosureKind::FnOnce => ty::IntTy::I32,
        };
        Self::new(interner, ty::Int(int_ty))
    }

    pub fn from_coroutine_closure_kind(interner: I, kind: ty::ClosureKind) -> Self {
        let int_ty = match kind {
            ty::ClosureKind::Fn | ty::ClosureKind::FnMut => ty::IntTy::I16,
            ty::ClosureKind::FnOnce => ty::IntTy::I32,
        };
        Self::new(interner, ty::Int(int_ty))
    }

    // ==== Below here is what used to be `compiler/rustc_middle/src/ty/sty.rs`
    // Type utilities L931
    #[inline(always)]
    pub fn flags(self) -> TypeFlags {
        self.0.flags
    }

    pub fn is_ty_var(self) -> bool {
        matches!(self.kind(), ty::Infer(ty::TyVar(_)))
    }

    pub fn is_ty_error(self) -> bool {
        matches!(self.kind(), ty::Error(_))
    }

    /// Returns `true` if this type is a floating point type.
    pub fn is_floating_point(self) -> bool {
        matches!(self.kind(), ty::Float(_) | ty::Infer(ty::FloatVar(_)))
    }

    #[inline]
    pub fn is_trait(self) -> bool {
        matches!(self.kind(), ty::Dynamic(_, _))
    }

    pub fn is_integral(self) -> bool {
        matches!(self.kind(), ty::Infer(ty::IntVar(_)) | ty::Int(_) | ty::Uint(_))
    }

    pub fn is_fn_ptr(self) -> bool {
        matches!(self.kind(), ty::FnPtr(..))
    }

    pub fn has_unsafe_fields(self) -> bool {
        match self.kind() {
            ty::Adt(adt_def, _) => adt_def.has_unsafe_fields(),
            _ => false,
        }
    }

    #[tracing::instrument(level = "trace", skip(interner))]
    pub fn fn_sig(self, interner: I) -> ty::Binder<I, ty::FnSig<I>> {
        self.kind().fn_sig(interner)
    }

    pub fn discriminant_ty(self, interner: I) -> Ty<I> {
        interner.ty_discriminant_ty(self)
    }

    pub fn is_known_rigid(self) -> bool {
        self.kind().is_known_rigid()
    }

    pub fn is_guaranteed_unsized_raw(self) -> bool {
        match self.kind() {
            ty::Dynamic(_, _) | ty::Slice(_) | ty::Str => true,
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Adt(_, _)
            | ty::Foreign(_)
            | ty::Array(_, _)
            | ty::Pat(_, _)
            | ty::RawPtr(_, _)
            | ty::Ref(_, _, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(_, _)
            | ty::UnsafeBinder(_)
            | ty::Closure(_, _)
            | ty::CoroutineClosure(_, _)
            | ty::Coroutine(_, _)
            | ty::CoroutineWitness(_, _)
            | ty::Never
            | ty::Tuple(_)
            | ty::Alias(_, _)
            | ty::Param(_)
            | ty::Bound(_, _)
            | ty::Placeholder(_)
            | ty::Infer(_)
            | ty::Error(_) => false,
        }
    }
}
