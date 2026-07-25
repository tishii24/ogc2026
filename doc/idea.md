large-reconstructのinsertを探索する
- 今はgreedyに挿入しているが、これを改良したい
- 残っているblockに対するinsertを試す
1. 全てのblockをremoveする
2. 残っているblockでbase-scheduleを作成する
3. base-scheduleに対して、removeしたブロックそれぞれを挿入できる情報を取得して、cacheしておく
4. (bay-id,x,y,entry-t)の組合せを探索する
  - ビームサーチ or 局所探索
  - 残っているblockに対してinsertできることは保証されているので、removeしたブロックの状態に対して干渉しないことを確かめれば良い
  - 場合によっては、全て配置してからinsert-greedyをした方が良いかも
  - 計算量が良くなりそう
  - 残っているblockの方が10倍程度多い
  - scheduleを二つに分ける、みたいな実装で十分？
