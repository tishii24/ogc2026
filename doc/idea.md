timelimitは長そうなので、近傍を大きくして、より広く探索することが必要そう
パラメータを調整して、randomnessを増やすことである程度スコアは改善できそうだが、根本的な改善を見つけたい
insert-candidateはbbox-right,bbox-topで評価する必要はなさそう
一方で、詰めて配置する価値はあることはわかっている
色々なorientationを試すことに価値はありそう
- bbox-right, bbox-topの上位k件からランダムに選ぶようにする
- orientationはshuffleして、色々なorientationを試せるようにしたい

スコアが停滞していることを検知したら or 探索がある程度進んだら、徐々に近傍を大きく・randomnessを上げたい
ただ、個々のパラメータをチューニングするというよりかは、一つのスカラーで近傍の範囲を定義して、それを引数に他のパラメータは定義されるようにしたい

large-reconstructのinsertを探索する
- 今はgreedyに挿入しているが、これを改良したい
- 残っているblockに対するinsertを試す
1. 全てのblockをremoveする
2. 残っているblockでbase-scheduleを作成する
3. base-scheduleに対して、removeしたブロックそれぞれを挿入できる情報を取得する
4. (bay-id,x,y,entry-t)の組合せをビームサーチで探索する

4の方法には以下のどれにするか悩んでいる
1. removed-blockごとに、base-scheduleへの挿入候補をC個用意して、それをビーム候補について試す
  - 各blockにつき、C個の候補は事前に生成しておく
  - ビーム中に追加したblockのscheduleに対しては、collisionの結果がcacheされていればO(1)で取得できるはず
  - 高速に判定できるが、候補が微妙だときついケースではうまくいかないかも
2. base-scheduleとビーム中に追加したblockのscheduleの結果をうまく結合して、ビーム候補を差分scan-yで探す
  - ビーム中に追加したblockをnew-scheduleとして、base-scheduleとnew-scheduleをうまく結合する
  - 高速に判定できれば嬉しいが、やり方はわからない
  - 高速に判定できれば、ビーム中に追加したblockに対しても正確にinsertを試せるため、複雑な挿入位置でも対応しやすい

いずれも、base-scheduleに含まれるblockの方が多いため、base-scheduleに対する挿入結果をうまく使い回すことで、計算量を改善したい
