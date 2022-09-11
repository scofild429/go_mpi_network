package network

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func (nn *NeuralNetFrame) FInputData2InputLayer(inputdata *mat.Dense, j int) {
	// 	fmt.Println(mat.Formatted(nn.wInput2InputLayer))
	nn.dInputLayerTmp.Mul(inputdata, nn.wInput2InputLayer)
	Input2InputLayerAddB := func(_, col int, v float64) float64 { return v + nn.bInput2InputLayer.At(0, col) }
	nn.dInputLayerTmp.Apply(Input2InputLayerAddB, nn.dInputLayerTmp)
}
func (nn *NeuralNetFrame) FInputLayer2HiddenLayer(j int) {
	nn.dHiddenLayerTmp.Mul(nn.dInputLayer, nn.wInputLayer2HiddenLayer)
	InputLayer2HiddenLayerAddB := func(_, col int, v float64) float64 { return v + nn.bInputLayer2HiddenLayer.At(0, col) }
	nn.dHiddenLayerTmp.Apply(InputLayer2HiddenLayerAddB, nn.dHiddenLayerTmp)
}
func (nn *NeuralNetFrame) FHiddenLayer2OutputLayer(j int) {
	nn.dOutputLayerTmp.Mul(nn.dHiddenLayer, nn.wHiddenLayer2OutLayer)
	HiddenLayer2OutputLayerAddB := func(_, col int, v float64) float64 { return v + nn.bHiddenLayer2OutLayer.At(0, col) }
	nn.dOutputLayerTmp.Apply(HiddenLayer2OutputLayerAddB, nn.dOutputLayerTmp)
}
func (nn *NeuralNetFrame) Output(outputlables *mat.Dense) float64 {
	nn.dOutputLayerErr.Sub(outputlables, nn.dOutputLayer)
	nn.dOutputLayerLoss.Apply(nn.Config.applyL2loss, nn.dOutputLayerErr)
	d := mat.Sum(nn.dOutputLayerLoss)
	return d
}
func (nn *NeuralNetFrame) Evaluation(outputlabels *mat.Dense) int64 {
	// fmt.Println(outputlabels)
	// fmt.Println(nn.dOutputLayer)
	maxdatavalue := nn.dOutputLayer.At(0, 0)
	maxdataindex := 0
	for j := 0; j < nn.Config.labelOnehotDims; j++ {
		if nn.dOutputLayer.At(0, j) > maxdatavalue {
			maxdatavalue = nn.dOutputLayer.At(0, j)
			maxdataindex = j
		}
	}
	for k := 0; k < nn.Config.labelOnehotDims; k++ {
		if outputlabels.At(0, k) == 1 && k == maxdataindex {
			return 1
		}
	}
	return 0
}
func (nn *NeuralNetFrame) BOutput2OutputLayer(j int) {
	nn.dOutputLayerAdj.MulElem(nn.dOutputLayerErr, nn.dOutputLayerSlope)
	nn.wHiddenLayer2OutLayerAdj.Mul(nn.dHiddenLayer.T(), nn.dOutputLayerAdj)
	nn.wHiddenLayer2OutLayerAdj.Scale(nn.Config.learningRate, nn.wHiddenLayer2OutLayerAdj)

	var err error
	nn.bHiddenLayer2OutLayerAdj, err = sumAlongAxis(0, nn.dOutputLayerAdj)
	if err != nil {
		fmt.Println("error comes", err)
	}
	nn.bHiddenLayer2OutLayerAdj.Scale(nn.Config.learningRate, nn.bHiddenLayer2OutLayerAdj)

}
func (nn *NeuralNetFrame) BOutputLayer2HiddenLayer(j int) {
	nn.dHiddenLayerErr.Mul(nn.wHiddenLayer2OutLayer, nn.dOutputLayerErr.T())
	nn.dHiddenLayerAdj.MulElem(nn.dHiddenLayerErr.T(), nn.dHiddenLayerSlope)
	nn.wInputLayer2HiddenLayerAdj.Mul(nn.dInputLayer.T(), nn.dHiddenLayerAdj)
	nn.wInputLayer2HiddenLayerAdj.Scale(nn.Config.learningRate, nn.wInputLayer2HiddenLayerAdj)

	var err error
	nn.bInputLayer2HiddenLayerAdj, err = sumAlongAxis(0, nn.dHiddenLayerAdj)
	if err != nil {
		fmt.Println("error comes", err)
	}
	nn.bInputLayer2HiddenLayerAdj.Scale(nn.Config.learningRate, nn.bInputLayer2HiddenLayerAdj)
}
func (nn *NeuralNetFrame) BHiddenLayer2InputLayer(inputdata *mat.Dense, j int) {
	nn.dInputLayerErr.Mul(nn.wInputLayer2HiddenLayer, nn.dHiddenLayerAdj.T())
	nn.dInputLayerAdj.MulElem(nn.dInputLayerErr.T(), nn.dInputLayerSlope)
	nn.wInput2InputLayerAdj.Mul(inputdata.T(), nn.dInputLayerAdj)
	nn.wInput2InputLayerAdj.Scale(nn.Config.learningRate, nn.wInput2InputLayerAdj)
	var err error
	nn.bInput2InputLayerAdj, err = sumAlongAxis(0, nn.dInputLayerAdj)
	if err != nil {
		fmt.Println("error comes", err)
	}
	nn.bInput2InputLayerAdj.Scale(nn.Config.learningRate, nn.bInput2InputLayerAdj)
}
func (nn *NeuralNetFrame) AccuWeightsMPI(j int) {
	nn.wInput2InputLayerMPI.Add(nn.wInput2InputLayerMPI, nn.wInput2InputLayerAdj)
	nn.bInput2InputLayerMPI.Add(nn.bInput2InputLayerMPI, nn.bInput2InputLayerAdj)
	nn.wInputLayer2HiddenLayerMPI.Add(nn.wInputLayer2HiddenLayerMPI, nn.wInputLayer2HiddenLayerAdj)
	nn.bInputLayer2HiddenLayerMPI.Add(nn.bInputLayer2HiddenLayerMPI, nn.bInputLayer2HiddenLayerAdj)
	nn.wHiddenLayer2OutLayerMPI.Add(nn.wHiddenLayer2OutLayerMPI, nn.wHiddenLayer2OutLayerAdj)
	nn.bHiddenLayer2OutLayerMPI.Add(nn.bHiddenLayer2OutLayerMPI, nn.bHiddenLayer2OutLayerAdj)
}

func (nn *NeuralNetFrame) UpdateWeightsSingleNode(j int) {
	nn.wInput2InputLayer.Add(nn.wInput2InputLayer, nn.wInput2InputLayerAdj)
	nn.bInput2InputLayer.Add(nn.bInput2InputLayer, nn.bInput2InputLayerAdj)
	nn.wInputLayer2HiddenLayer.Add(nn.wInputLayer2HiddenLayer, nn.wInputLayer2HiddenLayerAdj)
	nn.bInputLayer2HiddenLayer.Add(nn.bInputLayer2HiddenLayer, nn.bInputLayer2HiddenLayerAdj)
	nn.wHiddenLayer2OutLayer.Add(nn.wHiddenLayer2OutLayer, nn.wHiddenLayer2OutLayerAdj)
	nn.bHiddenLayer2OutLayer.Add(nn.bHiddenLayer2OutLayer, nn.bHiddenLayer2OutLayerAdj)
}

func (nn *NeuralNetFrame) InitialWeightsMPISendRecv(j int) {
	nn.wInput2InputLayerMPI.Zero()
	nn.bInput2InputLayerMPI.Zero()
	nn.wInputLayer2HiddenLayerMPI.Zero()
	nn.bInputLayer2HiddenLayerMPI.Zero()
	nn.wHiddenLayer2OutLayerMPI.Zero()
	nn.bHiddenLayer2OutLayerMPI.Zero()
}

func (nn *NeuralNetFrame) InitialWeightsMPIAllreduce(j int) {
	nn.wInput2InputLayerAdj.Zero()
	nn.bInput2InputLayerAdj.Zero()
	nn.wInputLayer2HiddenLayerAdj.Zero()
	nn.bInputLayer2HiddenLayerAdj.Zero()
	nn.wHiddenLayer2OutLayerAdj.Zero()
	nn.bHiddenLayer2OutLayerAdj.Zero()
}
