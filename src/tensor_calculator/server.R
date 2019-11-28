#
# This is the server logic of a Shiny web application. You can run the 
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)

source("calculate_output_size.R")

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
   
  output_size <- reactive({
    calculate_output_size(input = input$input_size, 
                          kernel = input$kernel_size, 
                          stride = input$stride, 
                          padding = input$padding, 
                          dilation = input$dilation, 
                          type_layer = input$type_layer,
                          output_padding = input$output_padding)
  })
  
  output$output_size <- renderText({
    as.character(output_size())
  })
  
})
