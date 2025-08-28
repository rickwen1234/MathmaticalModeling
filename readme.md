
# 项目目录结构

```plaintext
MathmaticalModeling/
  LICENSE
  requirements.txt
  readme.md
  四层结构拆分与项目内调用范式（python_方案）.md
  problem1_snapshots.csv
  result1_chain.csv
  result4_chain.xlsx
  src/
    apps/
      common_io.py         # 可视化/导出工具
      problem1.py          # 应用脚本：链仿真+导出
      problem2.py          # 应用脚本：参数判据与对比图
      problem3.py          # 应用脚本：最小螺距判据与对比图
      problem4.py          # 应用脚本：S弯+链仿真+导出
    services/
      geometry.py          # 几何原语/积分与反解
      path.py              # 路径构造（螺线+S弯）
      rigid_chain.py       # 刚性链解算器
      metrics/
        coords.py          # 坐标相关
        kinematics.py      # 运动学相关度量
      viz/
        heatmap.py         # 热力图绘制
        merge.py           # 图片整合
        trace.py           # 节点轨迹绘制
      __init__.py
    
  tests/
    # 当前为空，可添加测试脚本
```