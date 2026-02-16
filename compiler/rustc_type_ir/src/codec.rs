#![cfg(feature = "nightly")]

use std::hash::Hash;
use std::intrinsics;
use std::marker::DiscriminantKind;

use rustc_data_structures::fx::FxHashMap;
use rustc_serialize::{Decodable, Encodable};
use rustc_span::{SpanDecoder, SpanEncoder};

use crate::inherent::*;
use crate::{self as ty, Interner, Ty};

/// The shorthand encoding uses an enum's variant index `usize`
/// and is offset by this value so it never matches a real variant.
/// This offset is also chosen so that the first byte is never < 0x80.
pub const SHORTHAND_OFFSET: usize = 0x80;

pub trait TyEncoder<'tcx>: SpanEncoder {
    type Interner: Interner + 'tcx;

    const CLEAR_CROSS_CRATE: bool;

    fn position(&self) -> usize;

    fn type_shorthands(&mut self) -> &mut FxHashMap<Ty<Self::Interner>, usize>;

    fn predicate_shorthands(&mut self) -> &mut FxHashMap<ty::PredicateKind<Self::Interner>, usize>;

    fn encode_alloc_id(&mut self, alloc_id: &<Self::Interner as Interner>::AllocId);
}

pub trait TyDecoder<'tcx>: SpanDecoder {
    type Interner: Interner + 'tcx;

    const CLEAR_CROSS_CRATE: bool;

    fn interner(&self) -> Self::Interner;

    fn cached_ty_for_shorthand<F>(
        &mut self,
        shorthand: usize,
        or_insert_with: F,
    ) -> Ty<Self::Interner>
    where
        F: FnOnce(&mut Self) -> Ty<Self::Interner>;

    fn with_position<F, R>(&mut self, pos: usize, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R;

    fn positioned_at_shorthand(&self) -> bool {
        (self.peek_byte() & (SHORTHAND_OFFSET as u8)) != 0
    }

    fn decode_alloc_id(&mut self) -> <Self::Interner as Interner>::AllocId;
}

pub trait EncodableWithShorthand<'tcx, E: TyEncoder<'tcx>>: Copy + Eq + Hash {
    type Variant: Encodable<E> + Copy;
    fn variant(&self) -> Self::Variant;
}

impl<'tcx, I, E> EncodableWithShorthand<'tcx, E> for Ty<I>
where
    I: Interner,
    E: TyEncoder<'tcx, Interner = I>,
    ty::TyKind<I>: Encodable<E>,
{
    type Variant = ty::TyKind<I>;

    #[inline]
    fn variant(&self) -> Self::Variant {
        self.kind()
    }
}

impl<'tcx, I, E> EncodableWithShorthand<'tcx, E> for ty::PredicateKind<I>
where
    I: Interner,
    E: TyEncoder<'tcx, Interner = I>,
    ty::PredicateKind<I>: Encodable<E>,
{
    type Variant = ty::PredicateKind<I>;

    #[inline]
    fn variant(&self) -> Self::Variant {
        *self
    }
}

/// Encode the given value or a previously cached shorthand.
pub fn encode_with_shorthand<'tcx, E, T, M>(encoder: &mut E, value: &T, cache: M)
where
    E: TyEncoder<'tcx>,
    M: for<'b> Fn(&'b mut E) -> &'b mut FxHashMap<T, usize>,
    T: EncodableWithShorthand<'tcx, E>,
    // The discriminant and shorthand must have the same size.
    T::Variant: DiscriminantKind<Discriminant = isize>,
{
    let existing_shorthand = cache(encoder).get(value).copied();
    if let Some(shorthand) = existing_shorthand {
        encoder.emit_usize(shorthand);
        return;
    }

    let variant = value.variant();

    let start = encoder.position();
    variant.encode(encoder);
    let len = encoder.position() - start;

    // The shorthand encoding uses the same usize as the
    // discriminant, with an offset so they can't conflict.
    let discriminant = intrinsics::discriminant_value(&variant);
    assert!(SHORTHAND_OFFSET > discriminant as usize);

    let shorthand = start + SHORTHAND_OFFSET;

    // Get the number of bits that leb128 could fit
    // in the same space as the fully encoded type.
    let leb128_bits = len * 7;

    // Check that the shorthand is a not longer than the
    // full encoding itself, i.e., it's an obvious win.
    if leb128_bits >= 64 || (shorthand as u64) < (1 << leb128_bits) {
        cache(encoder).insert(*value, shorthand);
    }
}

impl<'tcx, I, E> Encodable<E> for Ty<I>
where
    I: Interner,
    E: TyEncoder<'tcx, Interner = I>,
    ty::TyKind<I>: Encodable<E>,
{
    fn encode(&self, e: &mut E) {
        encode_with_shorthand(e, self, TyEncoder::type_shorthands);
    }
}

impl<'tcx, I, D> Decodable<D> for Ty<I>
where
    I: Interner,
    D: TyDecoder<'tcx, Interner = I>,
    ty::TyKind<I>: Decodable<D>,
{
    #[allow(rustc::usage_of_ty_tykind)]
    fn decode(decoder: &mut D) -> Ty<I> {
        // Handle shorthands first, if we have a usize > 0x80.
        if decoder.positioned_at_shorthand() {
            let pos = decoder.read_usize();
            assert!(pos >= SHORTHAND_OFFSET);
            let shorthand = pos - SHORTHAND_OFFSET;

            decoder.cached_ty_for_shorthand(shorthand, |decoder| {
                decoder.with_position(shorthand, Ty::decode)
            })
        } else {
            let interner = decoder.interner();
            interner.mk_ty_from_kind(ty::TyKind::decode(decoder))
        }
    }
}
