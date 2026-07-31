## 小さいケース

特定の構造を目指すより、たくさん動かして良い構造を探索することが重要

- reheat
- randomnessを上げる
  - directionをrandomにする

## 大きいケース

- 高速化
- preoptの改善
  - 近似方法・パラメータ
- 近傍の精度向上
  - reconstruct-weightのpowerをつける
  - seedの選び方を増やす
    - target-bay/time-window remove
  - moveを小さいブロックに限らない

### reheat

- best解が更新されたら必ずexchangeする
- best解を取得して、一定iterationが経ったらbest解に戻る
- 同じbest解を取得するのが2回目以上だったら、温度を少し上げてreheatする
