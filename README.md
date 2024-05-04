# Markowitz Model
Python implementation of the markowitz model, given expected returns and covariance matrix.


```python
import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import cvxpy as cp
import yfinance as yf
import plotly.express as px
from markowitz import *
```

## Model Inputs
Assuming the expected returns are equal to the average of the historical returns is indeed erroneus. It suggests trend-stationary time series and we know that even if we accept the weiner process assumption about the prices, the parameters of the process can vary across different time windows. For simplicity and need for inputs to the model, we will use historical moments.


```python
get_history = lambda ticker: yf.Ticker(ticker).history("1y")["Close"].pct_change(1).to_numpy()[5:]

s1 = get_history("SASA.IS")
s2 = get_history("QUAGR.IS")
s3 = get_history("SASA.IS")
s4 = get_history("SAHOL.IS")
s5 = get_history("YKBNK.IS")
```


```python
R = np.array([s1.mean(),s2.mean(),s3.mean(),s4.mean(),s5.mean()]) # expected returns
C = np.cov(np.array([s1,s2,s3,s4,s5])) # covariance matrix
print("Expected Returns:\n",R)
print("Covariance matrix:\n",C) 
print("Correlation matrix:\n",np.corrcoef(np.array([s1,s2,s3,s4,s5])))
```

    Expected Returns:
     [ 0.0012344  -0.00158661  0.0012344   0.00397837  0.00508667]
    Covariance matrix:
     [[1.07643948e-03 2.57116599e-04 1.07643948e-03 3.37078851e-04
      2.94572780e-04]
     [2.57116599e-04 8.68971163e-04 2.57116599e-04 1.48168704e-04
      9.69006156e-05]
     [1.07643948e-03 2.57116599e-04 1.07643948e-03 3.37078851e-04
      2.94572780e-04]
     [3.37078851e-04 1.48168704e-04 3.37078851e-04 6.80297365e-04
      4.85946949e-04]
     [2.94572780e-04 9.69006156e-05 2.94572780e-04 4.85946949e-04
      9.40616492e-04]]
    Correlation matrix:
     [[1.         0.26584753 1.         0.3939012  0.29274636]
     [0.26584753 1.         0.26584753 0.19271007 0.10718094]
     [1.         0.26584753 1.         0.3939012  0.29274636]
     [0.3939012  0.19271007 0.3939012  1.         0.60748166]
     [0.29274636 0.10718094 0.29274636 0.60748166 1.        ]]
    


```python
w = optimize_rmin(R,C,0) # just minimize variance without regarding the return
w.value
```




    array([0.07195286, 0.35126476, 0.07195286, 0.31648497, 0.18834454])




```python
# plot the efficient frontier
plot_efficient_frontier(R,C,100)
plt.show()
```


    
![png](./images/output_6_0.png)
    



```python
# market portfolio when risk-free rate equals 0
find_market_portfolio(R,C,rf=0.0)
```




    array([0.        , 0.        , 0.        , 0.43895826, 0.56104174])




```python
# plot of the sharpe ratio and overall efficient frontier
rf = 0
plot_efficient_frontier(R,C,100)
w_best = find_market_portfolio(R,C,rf,100)
plt.plot([0,np.sqrt(w_best.T@C@w_best)],[rf,w_best@R],color="red")
plt.show()
print(f"sharpe for risk free overnigh {rf}:",(w_best@R-rf)/np.sqrt(w_best.T@C@w_best))
```


    
![png]([output_8_0.png](https://github.com/ramazanCevik/markowitz-model/blob/cad32bfb46b1c98596769ecb08c8cf8c20761d82/images/output_8_0.png))
    


    sharpe for risk free overnigh 0: 0.17820357496807757
    

## Effect of the Risk-Free Rate 
The risk free rate changes the market portfolio. Below, consider the changing risk-free rate as available (or mandatory for negative rates) by a single agent since the changing publicly available risk-free rate changes the expected returns and variances in real life.


```python
rf_set = np.linspace(0,max(R)-1e-8,30)
portfolio_set = [find_market_portfolio(R,C,rf,30,epsilon=-1e-7) for rf in rf_set] 
return_set = [x@R for x in portfolio_set]
std_set = [np.sqrt(x.T@C@x) for x in portfolio_set]
sharpe_set = [sharpe(x,rf,C,R) for x in portfolio_set]
```


```python
fig = px.line_3d(x=rf_set, y=std_set, z=return_set,
                 labels={"x":"risk-free rate","y":"std (without rf)","z":"return (without rf)"})
fig.show()
```


<div>                            <div id="17e084c2-c120-4155-a96d-d4ec9ddbf7a2" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("17e084c2-c120-4155-a96d-d4ec9ddbf7a2")) {                    Plotly.newPlot(                        "17e084c2-c120-4155-a96d-d4ec9ddbf7a2",                        [{"hovertemplate":"risk-free rate=%{x}<br>std (without rf)=%{y}<br>return (without rf)=%{z}<extra></extra>","legendgroup":"","line":{"color":"#636efa","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"","scene":"scene","showlegend":false,"x":[0.0,0.0001754021717499787,0.0003508043434999574,0.0005262065152499361,0.0007016086869999148,0.0008770108587498935,0.0010524130304998722,0.001227815202249851,0.0014032173739998296,0.0015786195457498083,0.001754021717499787,0.0019294238892497656,0.0021048260609997445,0.002280228232749723,0.002455630404499702,0.0026310325762496805,0.002806434747999659,0.002981836919749638,0.0031572390914996165,0.003332641263249595,0.003508043434999574,0.0036834456067495525,0.0038588477784995312,0.00403424995024951,0.004209652121999489,0.004385054293749468,0.004560456465499446,0.004735858637249425,0.004911260808999404,0.005086662980749382],"y":[0.02571972673860765,0.025787419776835994,0.025862311617384514,0.02594596731129351,0.026045465206961615,0.02615920668585166,0.026287690810449668,0.026442472988474595,0.026623209305423286,0.026835965515798785,0.027095878304358674,0.027408735014108617,0.027795418246980508,0.02827964943051568,0.028899374062950275,0.029698093472918047,0.03066417690787421,0.03066417690787421,0.03066417690787421,0.03066417690787421,0.03066417690787421,0.03066417690787421,0.03066417690787421,0.03066417690787421,0.03066417690787421,0.03066417690787421,0.03066417690787421,0.03066417690787421,0.03066417690787421,0.03066417690787421],"z":[0.004583347244515565,0.004595172961719796,0.004607742948954858,0.004621223969903152,0.004636586380700503,0.004653373189015057,0.00467148002634698,0.0046922632307331725,0.004715325316324719,0.0047410773402661475,0.004770838823425813,0.004804633604939959,0.004843931370033827,0.004890083060567593,0.0049452865228741694,0.005011570803915297,0.005086277080315414,0.005086277080315414,0.005086277080315414,0.005086277080315414,0.005086277080315414,0.005086277080315414,0.005086277080315414,0.005086277080315414,0.005086277080315414,0.005086277080315414,0.005086277080315414,0.005086277080315414,0.005086277080315414,0.005086277080315414],"type":"scatter3d"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"scene":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"xaxis":{"title":{"text":"risk-free rate"}},"yaxis":{"title":{"text":"std (without rf)"}},"zaxis":{"title":{"text":"return (without rf)"}}},"legend":{"tracegroupgap":0},"margin":{"t":60}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('17e084c2-c120-4155-a96d-d4ec9ddbf7a2');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>

