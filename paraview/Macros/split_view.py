from paraview.simple import *

layout1 = GetLayout() 

layout1.SplitHorizontal(0, 0.5) 

layout1.SplitVertical(1, 0.3333) 

layout1.SplitVertical(2, 0.3333) 

layout1.SplitVertical(4, 0.5) 

layout1.SplitVertical(6, 0.5) 
