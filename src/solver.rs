use crate::{precompute::CollisionPrecompute, *};

pub fn solve(problem: &Problem, _timelimit: f64) -> Result<Solution, String> {
    const C: i64 = 10;
    eprintln!("building collision precompute...");
    let pre = CollisionPrecompute::build(problem, C);
    eprintln!("elapsed: {:.4}", time::elapsed_seconds());
    todo!()
}
