package network

import (
	"gonum.org/v1/gonum/mat"
)

func (nn *NeuralNetFrame) PrepWeightToBoadcast(MPIDATA []float64) {
	wInput2InputLayerData := nn.wInput2InputLayer.RawMatrix().Data
	bInput2InputLayerData := nn.bInput2InputLayer.RawMatrix().Data
	wInputLayer2HiddenLayerData := nn.wInputLayer2HiddenLayer.RawMatrix().Data
	bInputLayer2HiddenLayerData := nn.bInputLayer2HiddenLayer.RawMatrix().Data
	wHiddenLayer2OutLayerData := nn.wHiddenLayer2OutLayer.RawMatrix().Data
	bHiddenLayer2OutLayerData := nn.bHiddenLayer2OutLayer.RawMatrix().Data

	LenwInput2InputLayer := nn.Config.inputdataDims * nn.Config.inputLayerNeurons
	LenbInput2InputLayer := nn.Config.inputLayerNeurons
	LenwInputLayer2HiddenLayer := nn.Config.inputLayerNeurons * nn.Config.hiddenLayerNeurons
	LenbInputLayer2HiddenLayer := nn.Config.hiddenLayerNeurons
	LenwHiddenLayer2OutLayer := nn.Config.hiddenLayerNeurons * nn.Config.outputLayerNeurons
	LenbHiddenLayer2OutLayer := nn.Config.outputLayerNeurons
	sumtag := 0
	for i := 0; i < LenwInput2InputLayer; i++ {
		MPIDATA[i] = wInput2InputLayerData[i]
	}
	sumtag += LenwInput2InputLayer
	for i := 0; i < LenbInput2InputLayer; i++ {
		MPIDATA[sumtag+i] = bInput2InputLayerData[i]
	}
	sumtag += LenbInput2InputLayer
	for i := 0; i < LenwInputLayer2HiddenLayer; i++ {
		MPIDATA[sumtag+i] = wInputLayer2HiddenLayerData[i]
	}
	sumtag += LenwInputLayer2HiddenLayer
	for i := 0; i < LenbInputLayer2HiddenLayer; i++ {
		MPIDATA[sumtag+i] = bInputLayer2HiddenLayerData[i]
	}
	sumtag += LenbInputLayer2HiddenLayer
	for i := 0; i < LenwHiddenLayer2OutLayer; i++ {
		MPIDATA[sumtag+i] = wHiddenLayer2OutLayerData[i]
	}
	sumtag += LenwHiddenLayer2OutLayer
	for i := 0; i < LenbHiddenLayer2OutLayer; i++ {
		MPIDATA[sumtag+i] = bHiddenLayer2OutLayerData[i]
	}
}

func (nn *NeuralNetFrame) ReciveInitialWeight(MPIDATA []float64, rank int) {
	LenwInput2InputLayer := nn.Config.inputdataDims * nn.Config.inputLayerNeurons
	LenbInput2InputLayer := nn.Config.inputLayerNeurons
	LenwInputLayer2HiddenLayer := nn.Config.inputLayerNeurons * nn.Config.hiddenLayerNeurons
	LenbInputLayer2HiddenLayer := nn.Config.hiddenLayerNeurons
	LenwHiddenLayer2OutLayer := nn.Config.hiddenLayerNeurons * nn.Config.outputLayerNeurons
	LenbHiddenLayer2OutLayer := nn.Config.outputLayerNeurons
	sumtag := 0

	wInput2InputLayerData := MPIDATA[:LenwInput2InputLayer]
	sumtag += LenwInput2InputLayer
	bInput2InputLayerData := MPIDATA[sumtag : LenbInput2InputLayer+sumtag]
	sumtag += LenbInput2InputLayer
	wInputLayer2HiddenLayerData := MPIDATA[sumtag : LenwInputLayer2HiddenLayer+sumtag]
	sumtag += LenwInputLayer2HiddenLayer
	bInputLayer2HiddenLayerData := MPIDATA[sumtag : LenbInputLayer2HiddenLayer+sumtag]
	sumtag += LenbInputLayer2HiddenLayer
	wHiddenLayer2OutLayerData := MPIDATA[sumtag : LenwHiddenLayer2OutLayer+sumtag]
	sumtag += LenwHiddenLayer2OutLayer
	bHiddenLayer2OutLayerData := MPIDATA[sumtag : LenbHiddenLayer2OutLayer+sumtag]

	nn.wInput2InputLayer = mat.NewDense(nn.Config.inputdataDims, nn.Config.inputLayerNeurons, wInput2InputLayerData)
	nn.bInput2InputLayer = mat.NewDense(1, nn.Config.inputLayerNeurons, bInput2InputLayerData)
	nn.wInputLayer2HiddenLayer = mat.NewDense(nn.Config.inputLayerNeurons, nn.Config.hiddenLayerNeurons, wInputLayer2HiddenLayerData)
	nn.bInputLayer2HiddenLayer = mat.NewDense(1, nn.Config.hiddenLayerNeurons, bInputLayer2HiddenLayerData)
	nn.wHiddenLayer2OutLayer = mat.NewDense(nn.Config.hiddenLayerNeurons, nn.Config.outputLayerNeurons, wHiddenLayer2OutLayerData)
	nn.bHiddenLayer2OutLayer = mat.NewDense(1, nn.Config.outputLayerNeurons, bHiddenLayer2OutLayerData)
	// fmt.Println("Parallel", rank, " Network is ", mat.Formatted(nn.bHiddenLayer2OutLayer), "for initialization")
}
