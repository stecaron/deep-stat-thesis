#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)

# Define UI for application that draws a histogram
shinyUI(fluidPage(
  
  # Application title
  titlePanel("Tensors size Calculator"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
      selectInput("type_layer", "Layer type:", choices = c("Conv2d", "MaxPool2d", "ConvTranspose2d"), selected = "Conv2d"),
      numericInput("input_size", "Input Size:", value = 28, min = 1, step = 1),
      numericInput("kernel_size", "Kernel Size:", value = 4, min = 1, step = 1),
      numericInput("stride", "Stride:", value = 1, min = 1, step = 1),
      numericInput("padding", "Padding:", value = 0, min = 0, step = 1),
      numericInput("dilation", "Dilatation:", value = 1, min = 0, step = 1)
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
       textOutput("output_size")
    )
  )
))
