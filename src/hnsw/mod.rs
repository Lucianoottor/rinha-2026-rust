mod distance;
mod build;
mod index;

pub use index::StaticHNSW;
pub use build::build_index;

#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

const NONE: u32 = u32::MAX;
