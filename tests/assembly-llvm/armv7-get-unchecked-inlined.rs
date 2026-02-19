//@ only-arm
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 --target armv7-unknown-linux-gnueabihf
//@ needs-llvm-components: arm
//@ ignore-backends: gcc
#![crate_type = "lib"]
#![feature(arm_target_feature)]
#![feature(stdarch_arm_neon_intrinsics)]

// CHECK-LABEL: get_unchecked_range:
// CHECK: add  r0, r0, #3
// CHECK: sub  r1, r1, #3
// CHECK: bx   lr
#[no_mangle]
#[target_feature(enable = "neon")]
pub unsafe fn get_unchecked_range(x: &[u8]) -> &[u8] {
    unsafe { x.get_unchecked(3..) }
}

// CHECK-LABEL: get_unchecked_one:
// CHECK: add  r0, r0, #3
// CHECK: bx   lr
#[target_feature(enable = "neon")]
pub unsafe fn get_unchecked_one(x: &[u8]) -> &u8 {
    unsafe { x.get_unchecked(3) }
}
