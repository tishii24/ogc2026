use crate::{
    collision::{BlockPlacement, CollisionResult},
    precompute::Precompute,
    *,
};
use std::collections::BTreeMap;

pub(crate) fn _can_insert(
    pre: &Precompute,
    new_block: ScheduledBlock,
    schedule: &[ScheduledBlock],
) -> bool {
    fn interval_overlaps(a0: i64, a1: i64, b0: i64, b1: i64) -> bool {
        a0 < b1 && b0 < a1
    }

    fn place(s: ScheduledBlock) -> BlockPlacement {
        BlockPlacement {
            block_id: s.block_id,
            orient_idx: s.orient_idx,
            x: s.x,
            y: s.y,
        }
    }

    fn crane_clear(pre: &Precompute, moving: BlockPlacement, fixed: BlockPlacement) -> bool {
        pre.collision.crane(moving, fixed) == CollisionResult::Clear
    }

    fn placements_clear(pre: &Precompute, new_block: ScheduledBlock, old: ScheduledBlock) -> bool {
        let new_place = place(new_block);
        let old_place = place(old);

        if new_block.entry_time < old.entry_time && old.exit_time < new_block.exit_time {
            crane_clear(pre, old_place, new_place)
        } else if old.entry_time < new_block.entry_time && new_block.exit_time < old.exit_time {
            crane_clear(pre, new_place, old_place)
        } else {
            // ABAB 型は両方向が必要。同時刻の ENTRY/EXIT も操作順に依存するため保守的に両方向を見る。
            crane_clear(pre, new_place, old_place) && crane_clear(pre, old_place, new_place)
        }
    }

    for old in schedule.iter().filter(|old| {
        old.bay_id == new_block.bay_id
            && interval_overlaps(
                old.entry_time,
                old.exit_time,
                new_block.entry_time,
                new_block.exit_time,
            )
    }) {
        if !placements_clear(pre, new_block, *old) {
            return false;
        }
    }
    true
}
