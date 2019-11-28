

calculate_output_size <- function(input, kernel, stride, padding, dilation, type_layer, output_padding=0){
  
  if (type_layer == "Conv2d"){
    floor(((input + (2 * padding) - dilation * (kernel - 1) - 1)/stride) + 1)
  } else if (type_layer == "MaxPool2d"){
    floor(((input + (2 * padding) - dilation * (kernel - 1) - 1)/stride) + 1)
  } else if (type_layer == "ConvTranspose2d"){
    (input - 1) * stride - (2 * padding) + dilation * (kernel - 1) + output_padding + 1
  } else {
    stop("This layer type is not implemented.")
  }
  
}