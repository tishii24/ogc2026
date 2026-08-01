- adaptive-annealing
- multi-stage
  - tardiness -> pref -> pref + loads
- preoptimizeの改善
  - 近似方法・パラメータ
- initial-build、constraint-optimizeの改善
  - bay割り当てをもっと重視する
  - constraint-optimizeはbay割当をしばらく固定する

## 理想的な状態に向かう

実行時間はあるので、目指している解の状態にはだいたい到達できると考えて良い
となると、目指す解の状態の決定と、目指し方を適切に定めてあげる必要がある

理想的な状態はpreoptimizeで得られると仮定する（この仮定は疑い、調整する必要はある）
理想的な状態に向かうにはどうすれば良いか？

1. multi-stage optimize
  - obj1,obj3,obj2のweight-schedule
2. constraint optimize
  - bay-assign,insert-priorityをconstraintとする
3. guided optimize
  - bay-assign,entry-tをguideとして評価項に入れる
  - bay-assignを求める必要がある

## adaptive-parallel-annealing

ケースによってスコアのスケールが異なり、適切な温度設定が異なる
最適化時間が長い
並列性を活かしたい
