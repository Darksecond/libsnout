#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EyeShape {
    LeftEyePitch,
    LeftEyeYaw,
    LeftEyeLid,
    RightEyePitch,
    RightEyeYaw,
    RightEyeLid,
}
