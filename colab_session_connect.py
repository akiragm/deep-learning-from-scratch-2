# �Z�b�V�������ێ����邽�߂̃R�[�h
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

# �o�͂��N���A���邽�߂̃R�[�h
output.clear()