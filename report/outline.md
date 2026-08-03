# Abstract

- 本問題は、各ブロックのベイ、向き、座標、搬入時刻、搬出時刻を、時間制約と多層形状の幾何制約の下で同時に決定する問題である。
- 提案法は、実幾何制約を時刻別の面積容量制約に緩和する抽象 preoptimization と、そこで得た搬入順序に従う rolling-horizon 最適化からなる。
- 各 rolling horizon では、空間走査と禁止搬入時刻区間を統合した貪欲挿入で実行可能解を構築し、large reconstruction を含む複数種類の近傍を用いた並列焼きなましで改善する。
- 密なパッキングでは初期配置から異なる構造へ移ることが難しいため、局所探索の前に良い挿入順序を求めることを重視した。
- TODO: 最終ラウンドでの主要な改善点と計算結果を追記する。

# 1. Introduction

## 1.1 Challenge Problem & Main Difficulties

- ブロック $i$ について、ベイ $b_i$、向き $o_i$、座標 $(x_i,y_i)$、搬入時刻 $e_i$、搬出時刻 $q_i$ を決定する。
- 時間制約は $e_i\ge R_i$ および $q_i-e_i\ge P_i$ であり、同じ時刻に同じベイを占有するブロックは多層ポリゴンの衝突制約を満たす必要がある。
- ENTRY/EXIT 時には、対象ブロックの上方を他のブロックが遮らないというクレーン制約も満たさなければならない。
- 目的関数は、総遅れ $Z_1$、作業量不均衡 $Z_2$、ベイ選好ペナルティ $Z_3$ の重み付き和
  $$F=w_1Z_1+w_2Z_2+w_3Z_3$$
  である。
- 幾何制約が強いため、密に敷き詰めた実行可能解から別の配置構造へ局所的に遷移することが難しく、初期解の構造が最終性能を大きく左右する。
- ブロックを一つ動かすと、その時刻以降に同じ領域を利用する多数のブロックとの関係が変化するため、特に早い時刻の配置ほど変更しづらい。

## 1.2 Contributions

- 実幾何制約を時刻別の面積容量制約へ緩和し、ベイ割当と搬入時刻を先に求める抽象 preoptimization を導入した。
- 抽象解の搬入時刻からブロックの挿入順序を定め、実幾何問題を rolling horizon で逐次構築する方式を採用した。
- 多層衝突の $x$ 方向走査とクレーン制約から導かれる禁止搬入時刻区間を統合し、空間と時間を同時に探索する貪欲挿入法を実装した。
- 小規模な移動に加え、関連する複数ブロックを除去・再挿入する large reconstruction を焼きなまし近傍として導入した。
- 複数ワーカー間の良解共有、reheating、tabu を組み合わせ、制限時間内で探索を並列化した。

# 2. Algorithm Description

## 2.1 Overall Algorithm Framework

### Key Findings and Design Rationale

- 総遅れは $Z_1=\sum_i\max(0,q_i-D_i)$ であり、ブロックの大きさによらず各ブロックの遅れが同じ重みで加算される。このため、小ブロック群を先に処理する方が総遅れを抑えやすい場合がある。
- 一方、大ブロックは空き領域が広い段階で配置した方が挿入しやすく、時間目的に有利な順序とパッキングに有利な順序は必ずしも一致しない。
- また、幾何制約とクレーン制約が強いため、密なスケジュールから異なる配置構造へ局所操作だけで移ることは難しい。したがって、初期配置を構成する挿入順序が最終性能を大きく左右する。

#### Abstract Preoptimization for Admission Ordering

- Rolling horizon を適用するには、どのブロックから部分問題へ追加するかを事前に決める必要がある。
- release time や due date による単純な順序は、ブロックサイズ、処理時間、ベイ容量、ベイ選好、および他ブロックとの競合を同時には考慮できない。
- そこで、正確な幾何制約を面積容量制約へ緩和した abstract preoptimization により、全ブロックのベイ割当 $\hat b_i$ と搬入時刻 $\hat e_i$ を先に最適化する。
- $\hat e_i$ の昇順を用いることで、単純な release-time 順や due-date 順よりも、全体の tardiness を効率よく小さくできる挿入順序を得ることを目的とする。
- Abstract preoptimization は「どのブロックを先に挿入すべきか」を大域的に決め、後段の rolling horizon はその順序を実幾何配置へ変換する役割を持つ。

#### Effectiveness of Rolling-Horizon Optimization

- Rolling horizon では、搬入順序の前方からブロックを追加し、配置済みブロック数を限定した部分問題を逐次最適化する。
- 多くのケースで支配的な $Z_1$ と $Z_3$ は
  $$Z_1=\sum_i T_i,\qquad Z_3=\sum_i(S_i^{\max}-S_{i,b_i})$$
  とブロックごとの寄与に分解できる。幾何的実行可能性はブロック間で結合しているが、部分問題内の $Z_1,Z_3$ の改善は元の全体目的の改善と整合しやすい。
- 早い時刻のブロックは多数の後続ブロックと関係するため移動しづらい一方、スケジュール末尾のブロックには後続ブロックがなく、搬入時刻や配置を変更しやすい。
- Rolling horizon では、各ブロックが追加された horizon において一度はスケジュール末尾に位置するため、自由度が高い状態で良い配置を探索できる。
- 近傍評価の計算量は配置済みブロック数とともに増加する。前半を小さな部分問題として解くことで、最初から全体問題を探索し続ける場合よりも単位時間当たりの iteration 数を増やせる。
- 過去の horizon のブロックは固定せず、追加済みの全ブロックを探索対象に残す。これにより、部分問題化の利点を得ながら、後続ブロックの追加に応じた配置修正も許容する。

#### Large Reconstruction for Structural Changes

- 密なスケジュールでは、あるブロックの望ましい位置が周囲のブロックに占有されているため、一ブロックだけを動かしても大きな改善を得にくい。
- クレーン制約による時間的依存関係もあり、あるブロックの tardiness を改善するには、その周囲または前後にある複数ブロックを同時に変更する必要がある。
- そのため、Shift、Move、Rotate、Swap のような小規模近傍だけでは、現在のパッキング構造が作る局所最適解を越えることが難しい。
- Large reconstruction では、時間・空間的に関連する複数ブロックを一度取り除き、異なる順序で貪欲に再挿入することで、配置構造をまとめて変更する。
- 除去対象は目的関数への悪影響に加え、ENTRY時刻、空間的距離、slack、ベイ選好の自由度などから選び、無関係なブロックを除去する割合を抑える。
- 再挿入順序では、ブロックサイズ、期限の緊急度、slack、現在のペナルティなどを考慮し、tardiness と packability のトレードオフを再探索する。
- これにより、各候補スケジュールの実行可能性を維持しながら、単一ブロック近傍では到達しにくい異なるパッキング構造へ遷移する。

以上の三つのキーアイディアは、abstract preoptimization が大域的な挿入順序を決め、rolling horizon が小さく可動性の高い部分問題を構成し、large reconstruction が各部分問題内の配置構造を大きく組み替えるという役割を持つ。

### Solution Procedure

1. 各向きの境界矩形、占有形状、配置可能範囲、衝突情報、回転・交換候補を事前計算する。
2. 面積容量に緩和した抽象問題を解き、各ブロックのベイ $\hat b_i$ と搬入時刻 $\hat e_i$ を求める。
3. $(\hat e_i,D_i,i)$ の辞書順でブロックを並べ、同じ $\hat e_i$ を分断しないように rolling horizon を構成する。
4. horizon に新しく含まれるブロックを、実幾何制約を満たす位置へ貪欲に挿入する。
5. それまでに挿入した全ブロックを対象として焼きなましを行い、次の horizon へ進む。
6. 最後の horizon では $Z_1,Z_2,Z_3$ のすべてを用いて最終解を改善する。

## 2.2 Key Ideas that Succeeded

### 2.2.1 Abstract Preoptimization

- 実際の多層ポリゴン衝突を直接扱わず、ブロック $i$ をベイ $j$ に置く際の近似占有量 $a_{ij}$ を用いる。
- 向き $o$ の近似占有量を
  $$a_{ijo}=A^{\mathrm{union}}_{io}+\alpha\left(A^{\mathrm{bbox}}_{io}-A^{\mathrm{union}}_{io}\right)+\beta L_{io}$$
  とし、ベイに収まる向きの中で最小の値を $a_{ij}$ とする。ここで $L_{io}$ は外周長である。
- union area は実際の占有面積を表し、bounding-box gap は凹形状や細長い形状の周囲に生じる利用しづらい空間を近似する。perimeter 項は輪郭が複雑な形状のパッキング困難性を補正する。
- 各ベイ $j$、時刻 $t$ に対し、抽象容量制約
  $$\sum_{i:\,\hat e_i\le t<\hat e_i+P_i,\,\hat b_i=j}a_{ij}\le C_j$$
  を課す。$C_j$ はベイ面積から余白を除いた容量である。
- 抽象目的関数には公式目的関数に加えて、同時占有の集中を抑える小さな congestion penalty を加える。これは面積容量上は feasible でも、実際には配置困難となる高密度な抽象解を避けるためである。
- ランダム化した複数の挿入順序から初期解を構築し、relocate、swap、large reconstruction による焼きなましで改善する。
- この段階では正確な座標を決めず、全体の tardiness を小さくできるベイ割当と搬入順序を低コストで探索する。

### 2.2.2 Rolling-Horizon Geometric Optimization

- 抽象解の $\hat e_i$ が小さい順に一定数ずつブロックを追加し、rolling-horizon heuristic [2] と同様に部分問題を段階的に拡大する。
- 早い時刻の配置ほど後続ブロックへの影響が大きいため、前方から局所的に密度の高い探索を行う。
- 支配的な $Z_1$ と $Z_3$ は
  $$Z_1=\sum_i T_i,\qquad Z_3=\sum_i(S_i^{\max}-S_{i,b_i})$$
  のようにブロック単位に分解できるため、この逐次最適化でも全体目的との整合性を保ちやすい。
- 過去の horizon を完全には固定せず、追加済みの全ブロックを焼きなまし対象に残すことで、実幾何制約に応じた修正を許容する。
- $Z_2$ は全ブロックの割当が揃う前には評価が不安定なため、中間 horizon では重みをゼロとし、最後の horizon で有効化する。

### 2.2.3 Greedy Geometric Insertion

- 挿入時にはベイ、向き、$y$ 座標を列挙し、衝突イベントに基づく $x$ 方向走査から実行可能な座標と時刻を求める。
- 候補 $s$ は主に目的関数増分
  $$\Delta F(s)=w_1T_i(s)+w_2\Delta Z_2(s)+w_3(S_i^{\max}-S_{i,b_i(s)})$$
  で比較し、同点付近では早い搬入時刻とベイの隅へ寄せる配置を優先する。
- 常に左下へ寄せるのではなく、一定確率で他の三隅をアンカーとして選び、異なるパッキング構造を生成する。
- 多層の常時衝突制約と ENTRY/EXIT のクレーン制約をともに検査し、常に実行可能なスケジュールを維持する。

#### Collision Detection

- 各非凸レイヤーを ear clipping により凸多角形へ分解し、凸部分の組ごとに Minkowski difference [3] を計算する。
- 二つの凸多角形 $A,B$ に対し、相対変位 $(\Delta x,\Delta y)$ が
  $$A\oplus(-B)=\{a-b\mid a\in A,\ b\in B\}$$
  に含まれるとき両者は衝突する。この領域を整数 $\Delta y$ ごとに水平切断し、禁止される整数 $\Delta x$ の区間列として保存する。
- 数値誤差で衝突を見逃さないように小さな margin を加え、同じ行の重複・隣接区間をマージする。
- クレーン制約では、移動ブロックのレイヤー $k$ と固定ブロックのレイヤー $l\ge k$ の禁止領域を合併する。
- クレーン制約は移動方向に対して非対称であるため、ブロック対の両方向について別々の禁止区間を保持する。
- この禁止相対配置は、不規則形状パッキングにおける no-fit polygon [4] と同様の考え方である。
- 各ブロック・向き対の CollisionGrid は初めて必要になった時点で構築し、以降の探索ではキャッシュを再利用する。

#### Computational Complexity of Greedy Insertion

- ベイ $j$ に既に配置されたブロック数を $n_j$、ベイの幅と高さを $W_j,H_j$、挿入ブロックの向き数を $O_i$ とする。
- 各向きと整数 $y$ について、既存ブロックとの禁止 $x$ 区間から開始・終了イベントを生成し、counting sort と sweep を行う。
- ブロック対当たりの衝突区間数を定数とみなすと、イベント生成とソートは $O(n_j+W_j)$、各候補位置での禁止時刻区間の走査を含む保守的な上界は $O(W_j+n_j^2)$ である。
- したがって、一ブロックの greedy insertion の最悪計算量を概ね
  $$O\left(\sum_j O_iH_j(W_j+n_j^2)\right)$$
  と評価できる。
- 搬入可能時刻は日ごとに列挙せず、既存ブロックから導かれる禁止時刻区間をマージして最早実行可能時刻を求める。このため、計算量は時間 horizon に直接比例しない。

#### Joint Spatial-Temporal Sweep

- 各既存ブロックに対し、新ブロックが既存ブロックに遮られる方向 `new-to-old` と、既存ブロックが新ブロックに遮られる方向 `old-to-new` を別々に管理する。
- $x$ 方向のイベント走査中に二方向の干渉状態を更新し、その状態と既存ブロックの滞在区間から、新ブロックの禁止搬入時刻を高々二つの区間として導く。
- 現在の座標で有効な禁止時刻区間をマージし、探索範囲内の最早 feasible entry time を求める。これにより、各座標・各時刻でスケジュール全体を再検証することなく、位置と時刻を同時に決定する。

### 2.2.4 Simulated Annealing Neighborhoods

- 局所探索の基本枠組みとして simulated annealing [1] を用いる。
- Shift: 同じベイ・向きのまま近傍の $y$ 座標へ移し、より早い時刻に置ける位置を探す。
- Move: 小さなブロックを中心に選び、ベイ、向き、位置、時刻を貪欲挿入で決め直す。
- Rotate: 幾何的に近い向きと近傍座標を探索する。
- Swap: 形状・面積が近い二ブロックの位置を交換し、それぞれを再配置する。
- Large reconstruction: 遅れや選好が悪いブロック、時間・空間的に近いブロックなどを複数除去し、サイズ、期限、slack、現在のペナルティを考慮した順序で再挿入する。
- Large reconstruction の seed は badness、fluidity、random の複数方式から選び、各 seed と同じベイ内で $(x,y,t)$ 距離が近いブロックを集める。複数 seed を用いることで、複数の局所構造を一度に変更する。
- 再挿入順序を決めるサイズ、期限、slack、選好などの重みは試行ごとにランダム化し、固定順序では得られない多様なパッキング構造を探索する。
- 上記の構造化された除去戦略は実幾何最適化で用い、abstract preoptimization 内の large reconstruction ではより単純な複数ブロックの除去・再配置を用いる。
- 正の tardiness が残る局面と tardiness がゼロの局面で温度設定を切り替え、目的の探索尺度に合わせる。

## 2.3 Key Ideas that Failed

- Beam-search reconstruction: 同じ base schedule に対する挿入結果を複数の探索状態で再利用し、計算量を抑えながら再挿入順序を探索する方法を試した。
- しかし、挿入結果を使い回すよりも、多様な配置状態そのものを探索する方が重要であり、最終的には採用しなかった。
- Multi-stage optimization: tardiness、preference、workload imbalance を段階的に最適化する方法を試したが、通常の探索に対する優位な改善は確認できなかった。
- Adaptive annealing: 観測した目的関数差に応じて温度を適応的に調整する方法には可能性があったが、安定した改善を得るための調整が難しかった。
- Blocker-aware removal: クレーン作業を妨げるブロックを明示的に特定して reconstruction の除去対象とする方法にも可能性があったが、除去範囲や選択強度の調整が難しかった。

## 2.4 Further Improvement Plan

- 抽象 preoptimization の近似占有量とパラメータを改善し、抽象解と実幾何配置の乖離を小さくする。
- horizon サイズや探索時間の配分を改善し、前半の構造形成と最終改善のバランスを取る。
- 同時刻の ENTRY/EXIT の操作順序と、搬出時刻の変更によるブロック間の時間的順序の入れ替えを探索に取り入れる。
- 近傍操作で観測された目的関数差や受理率から、インスタンスと探索段階に応じて温度を自動調整する adaptive annealing を再検討する。
- ENTRY/EXIT を妨げる blocker と、その依存関係を利用して reconstruction の除去対象を選ぶ blocker-aware removal を改善する。
- 現在の複数ワーカーによる良解共有を拡張し、ワーカーごとに異なる温度帯や近傍構成を担当させ、一定間隔で状態を交換する並列焼きなまし手法を導入する。これにより、高温探索による配置構造の変更と低温探索による局所改善を両立する。

# 3. Implementation Details

- Rust で実装し、幾何演算には多角形の和領域、境界矩形、レイヤー間衝突の事前計算を利用する。
- 配置走査では、既存ブロックとの衝突が変化する $x$ 座標を利用し、全座標の総当たりを避ける。
- 抽象初期解の構築と焼きなましを複数ワーカーで並列実行する。
- ワーカー間で一定間隔ごとに良解を共有し、停滞時には best return と reheating を行う。
- 同一スケジュールの再訪を抑えるため、スケジュールのハッシュに基づく tabu を使用する。
- 制限時間を抽象 preoptimization と rolling-horizon 最適化へ配分し、後者では horizon の進行に応じて探索時間を割り当てる。
- 出力時には同一時刻の EXIT 操作をすべての ENTRY 操作より前に並べ、問題で定められた操作順序を保証する。
- Rust solver の外側の Python wrapper で、探索中に出力された候補解を目的値順に検証する。Shapely を用いて、割当・時間制約、ベイ境界、多層衝突、クレーン操作の実行可能性を独立に確認する。
- 最良候補が infeasible の場合は、以前に出力された次順位の feasible な候補へ fallback する。検証時間と返却時間は solver の制限時間からあらかじめ確保する。

# 4. Computational Results

- TODO: training dataset における $Z_1$、$Z_2$、$Z_3$、総目的値を表で掲載する。
- TODO: 各提出バージョンの目的値推移を掲載する。
- TODO: release-time 順、due-date 順、abstract preoptimization 順を比較し、挿入順序による $Z_1$ と総目的値の違いを掲載する。
- TODO: abstract preoptimization、rolling horizon、large reconstruction を順に追加したアブレーション結果を掲載する。
- TODO: rolling horizon の有無による単位時間当たりの iteration 数と目的値の違いを掲載する。
- TODO: 各インスタンスの目的関数の重みと $w_1Z_1,w_2Z_2,w_3Z_3$ の内訳を示し、$Z_1,Z_3$ が支配的となるケースを確認する。
- TODO: 結果から確認できる主要な傾向と限界を記述する。

# 5. Team Members' Contributions (Not Applicable to Single-Member Teams)

- Single-member team のため該当なしとする。

# 6. AI Use Declaration

- 本プロジェクトでは、主に Codex の GPT 5.6 Sol High を、著者が考案したアルゴリズムの実装、コードの読解・修正、デバッグの支援に使用した。
- 技術レポートの作成では、既存実装の整理、日本語の骨子と下書きの作成、文章校正、英語への翻訳に使用した。
- アルゴリズム上のアイデアと設計判断は著者が行い、生成されたコードと文章についても著者がレビュー、検証、修正した。

# References

1. S. Kirkpatrick, C. D. Gelatt, Jr., and M. P. Vecchi, “Optimization by Simulated Annealing,” Science, vol. 220, no. 4598, pp. 671–680, 1983. DOI: 10.1126/science.220.4598.671.
2. M. Singer, “Decomposition Methods for Large Job Shops,” Computers & Operations Research, vol. 28, no. 3, pp. 193–207, 2001. DOI: 10.1016/S0305-0548(99)00098-2.
3. P. K. Ghosh, “A Unified Computational Framework for Minkowski Operations,” Computers & Graphics, vol. 17, no. 4, pp. 357–378, 1993. DOI: 10.1016/0097-8493(93)90023-3.
4. E. K. Burke, R. S. R. Hellier, G. Kendall, and G. Whitwell, “A Comprehensive and Robust Procedure for Obtaining the No-Fit Polygon Using Minkowski Sums,” Computers & Operations Research, vol. 35, no. 1, pp. 267–281, 2008. DOI: 10.1016/j.cor.2006.02.026.
