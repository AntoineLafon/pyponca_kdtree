# pyponca_kdtree

Python bindings for the Ponca spatial-partitioning code (from [Ponca](https://github.com/poncateam/ponca) )
## How to build

To build the lib in an existing environment:

```bash
pip install pybind11 scikit-build-core
pip install -e . --no-build-isolation
```

After installation, verify that the module imports:

```bash
python -c "from ponca_kdtree import KdTree, KnnGraph; print(KdTree, KnnGraph)"
```


### small benchmark result (from [bench.py](./scripts/bench.py))

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-style: italic">                                       Benchmark Results                                       </span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">┃</span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Lib            </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">┃</span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Mean Build Time (s) </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">┃</span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Mean Radius Query Time (s) </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">┃</span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Mean KNN Query Time (s) </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">┃</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span><span style="font-weight: bold"> Ponca KDTree   </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>        <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">0.0089 </span><span style="color: #7fbf7f; text-decoration-color: #7fbf7f; font-weight: bold">1.00x</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>               <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">0.0220 </span><span style="color: #7fbf7f; text-decoration-color: #7fbf7f; font-weight: bold">1.00x</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>            0.0003 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">3.73x</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span><span style="font-weight: bold"> Ponca KNNGraph </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>       0.1636 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">18.44x</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>              1.6822 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">76.31x</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>            <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">0.0001 </span><span style="color: #7fbf7f; text-decoration-color: #7fbf7f; font-weight: bold">1.00x</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span><span style="font-weight: bold"> SciPy KDTree   </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>        0.0248 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">2.80x</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>               0.1325 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">6.01x</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>           0.0013 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">19.10x</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span><span style="font-weight: bold"> sklearn KDTree </span><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>        0.0424 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">4.78x</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>               0.0358 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1.62x</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>           0.0030 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">43.33x</span> <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│</span>
<span style="color: #7f7f7f; text-decoration-color: #7f7f7f">└────────────────┴─────────────────────┴────────────────────────────┴─────────────────────────┘</span>
</pre>