# 目录结构

```plaintext
project/
  pyproject.toml        # 或 requirements.txt
  src/
    services/
      geometry.py       # 第1层：几何原语/积分与反解（纯函数）
      path.py           # 第2层：路径构造（螺线+S弯），仅依赖 geometry
      rigid_chain.py    # 第3层：刚性链解算器（只依赖第2层的 path.eval）
    apps/
      common_io.py      # 第4层：可视化/导出工具（matplotlib+pandas）
      problem1.py       # 应用脚本：调用 1-3 层 + 导出
      problem3.py       # 应用脚本：最小螺距判据与对比图
      problem4.py       # 应用脚本：S弯+链仿真+导出
  tests/
    test_geometry.py
    test_path.py
    test_rigid_chain.py
  README.md
```