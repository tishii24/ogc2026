## 理想的な状態に向かう

実行時間は十分あるので、目指している解の状態にはだいたい到達できると考えて良い（特に大きいケースでは）
となると、目指す解の状態の決定と、目指し方を適切に定めてあげる必要がある

理想的な状態はpreoptimizeで得られると仮定する（この仮定は疑い、調整する必要はある）
理想的な状態に向かうにはどうすれば良いか？

1. multi-stage optimize
  - obj1,obj3,obj2のweight-schedule
2. constraint optimize
  - bay-assign,insert-priorityをconstraintとする
3. guided optimize
  - bay-assign,entry-tをguideとして評価項に入れる
  - 適切なbay-assignを求める必要があり、幾何制約を無視すると難しそう

小さいケースでは、上記に加えて、幾何制約に対するより良い配置を探索する必要がある

## adaptive-parallel-annealing

ケースによってスコアのスケールが異なり、適切な温度設定が異なる
最適化時間が長い
並列性を活かしたい

## multi-stage optimize

obj1->obj1+obj2->obj1+obj2+obj3 の順で最適化することを考えたい

- RawScore { z1, z2, z3 } を作る
- {obj1|obj2|obj3}-scaleのスケジュールを設定する
  - scaleは0->1にlinearにあげるようにする
  - パラメータとして、globalの最適化時間における{obj2|obj3}-scaleをあげるstart,endを設定できるようにする
  - z1が0になったらobj2-scaleを1にする
- annealing-scoreにはraw-scoreに重みをかけて計算する
- global-cはobj1だけを考慮する
- global-cはz1=0になったら終了する
