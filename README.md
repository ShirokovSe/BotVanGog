# BotVanGog
This TelegramBot is written with python and aiogram. This bot uses in basement different neural networks for different functions. He is able to transfer style from one picrute in another, classify buildings and paintings and even generate new paintings in demand.
# Main functions
## Style transfer
The main feature is a style transfer. For current realisation was choosed Hyang and Belongi suggestion about using Adain (Adaptive instance normalization). This idea allows to develop arbitrary style networks. And it works well. In ratio *quality/speed* this approach showed the best result. Here is an example of work:
<table align ="center">
  <tr><th>Style</th><th>Content</th><th>Transform</th>
  <tr>
    <td>
    <img src="https://github.com/ShirokovSe/BotVanGog/blob/main/Example/style.png" width="150">
    </td>
   <td>
   <img src="https://github.com/ShirokovSe/BotVanGog/blob/main/Example/content.png" width="150"</td>
  </td>
   <td>
   <img src="https://github.com/ShirokovSe/BotVanGog/blob/main/Example/transform.png" width="150"</td>
   </tr>
</table>
  
As you can see, suggested approach works with a good quality and it takes near 7-8 seconds on CPU to process it.

## Image generation
The next finction is generation paintings on demand. You can choose 1 out 9 different styles and bot send you collage from 9 paintings. Here is a few examples:
<table align ="center">
  <tr><th>Abstractionism</th><th>Romantism</th><th>Cubism</th><th>Pop Art</th>
  <tr>
    <td>
    <img src="https://github.com/ShirokovSe/BotVanGog/blob/main/Example/generated.png" width="150">
    </td>
   <td>
   <img src="https://github.com/ShirokovSe/BotVanGog/blob/main/Example/generated2.png" width="150"</td>
  </td>
   <td>
   <img src="https://github.com/ShirokovSe/BotVanGog/blob/main/Example/generated3.png" width="150"
        </td>
    <td>
   <img src="https://github.com/ShirokovSe/BotVanGog/blob/main/Example/generated4.png" width="150"
        </td>
   </tr>
</table>

  
 ## Buildings and paintings classification
 And the last functions are buildings and images classification. This Bot knows 5 buildings style and 10 paintings style. For buildings calssification was choosen and further trained pretrained ResNet50 , for paintings classification was chosen and further trainted pretrained Efficientb7. For buildings classification was gotten accuracy near 84%, for image classification near 75%. For my opinion, these result are quite good as sometimes it's very difficult to differ one style from another even for men.
