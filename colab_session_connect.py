# セッションを維持するためのコード
import IPython
from google.colab import output

display(IPython.display.Javascript('''
  function ClickConnect(){
    btn = document.querySelector("colab-connect-button")
    if (btn != null){
      console.log("Click colab-connect-button"); 
      btn.click() 
    }
  } 

  setInterval(ClickConnect,60000)
'''))

# 出力をクリアするためのコード
output.clear()