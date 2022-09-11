package network

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func (nn *NeuralNetFrame) ReviewNeteWork() {
	// 	fmt.Printf("%p in review\n", nn)
	// f := mat.Formatted(nn.wInput2InputLayer, mat.Prefix("                      "))
	// fmt.Printf("\nweight Input Layer  = %v\n", f)
	// f = mat.Formatted(nn.bInput2InputLayer, mat.Prefix("        "))
	// fmt.Printf("\nbais of Input Layer = %v\n\n\n", f)

	// f = mat.Formatted(nn.wInputLayer2HiddenLayer, mat.Prefix("                       "))
	// fmt.Printf("\nweight Hidden Layer  = %v\n", f)
	// f = mat.Formatted(nn.bInputLayer2HiddenLayer, mat.Prefix("        "))
	// fmt.Printf("\nbais of Hidden Layer = %v\n\n\n", f)

	// f := mat.Formatted(nn.wHiddenLayer2OutLayer, mat.Prefix("                      "))
	// fmt.Printf("\nweight Output Layer = %v\n", f)
	f := mat.Formatted(nn.bHiddenLayer2OutLayer, mat.Prefix("        "))
	fmt.Printf("\nbais of Output Layer = %v\n\n\n", f)

	// f := mat.Formatted(nn.dOutputLayer, mat.Prefix("                 "))
	// fmt.Printf("\nNetwork Output = %v\n\n\n", f)
}

func (nn *NeuralNetFrame) ReviewNeteWorkDims() {

	r, c := nn.wInput2InputLayer.Dims()
	fmt.Printf("\nweight Input Layer  havs %v row and %v colomns\n", r, c)

	r, c = nn.bInput2InputLayer.Dims()
	fmt.Printf("\nbais of Input Layer havs %v row and %v colomns\n", r, c)

	r, c = nn.wInputLayer2HiddenLayer.Dims()
	fmt.Printf("\nweight Hidden Layer havs %v row and %v colomns\n", r, c)

	r, c = nn.bInputLayer2HiddenLayer.Dims()
	fmt.Printf("\nbais of Hidden Layer havs %v row and %v colomns\n", r, c)

	r, c = nn.wHiddenLayer2OutLayer.Dims()
	fmt.Printf("\nweight Output Layer havs %v row and %v colomns\n", r, c)

	r, c = nn.bHiddenLayer2OutLayer.Dims()
	fmt.Printf("\nbais of Output Layer havs %v row and %v colomns\n", r, c)

	r, c = nn.dOutputLayer.Dims()
	fmt.Printf("\nNetwork Output havs %v row and %v colomns\n", r, c)
}
