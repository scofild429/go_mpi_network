package network

import (
	"gonum.org/v1/gonum/mat"
)

func (nn *NeuralNetFrame) PrepSendAdj(i int, j int, rank int) []float64 {
	wInput2InputLayerData := nn.wInput2InputLayerMPI.RawMatrix().Data
	bInput2InputLayerData := nn.bInput2InputLayerMPI.RawMatrix().Data
	wInputLayer2HiddenLayerData := nn.wInputLayer2HiddenLayerMPI.RawMatrix().Data
	bInputLayer2HiddenLayerData := nn.bInputLayer2HiddenLayerMPI.RawMatrix().Data
	wHiddenLayer2OutLayerData := nn.wHiddenLayer2OutLayerMPI.RawMatrix().Data
	bHiddenLayer2OutLayerData := nn.bHiddenLayer2OutLayerMPI.RawMatrix().Data

	LenwInput2InputLayer := nn.Config.inputdataDims * nn.Config.inputLayerNeurons
	LenbInput2InputLayer := nn.Config.inputLayerNeurons
	LenwInputLayer2HiddenLayer := nn.Config.inputLayerNeurons * nn.Config.hiddenLayerNeurons
	LenbInputLayer2HiddenLayer := nn.Config.hiddenLayerNeurons
	LenwHiddenLayer2OutLayer := nn.Config.hiddenLayerNeurons * nn.Config.outputLayerNeurons
	LenbHiddenLayer2OutLayer := nn.Config.outputLayerNeurons

	MPIDATA := make([]float64, LenwInput2InputLayer+LenbInput2InputLayer+LenwInputLayer2HiddenLayer+LenbInputLayer2HiddenLayer+LenwHiddenLayer2OutLayer+LenbHiddenLayer2OutLayer)
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

	return MPIDATA
}

func (nn *NeuralNetFrame) UpdateWeightsMain(AdjTmp []float64, rank int) {
	LenwInput2InputLayer := nn.Config.inputdataDims * nn.Config.inputLayerNeurons
	LenbInput2InputLayer := nn.Config.inputLayerNeurons
	LenwInputLayer2HiddenLayer := nn.Config.inputLayerNeurons * nn.Config.hiddenLayerNeurons
	LenbInputLayer2HiddenLayer := nn.Config.hiddenLayerNeurons
	LenwHiddenLayer2OutLayer := nn.Config.hiddenLayerNeurons * nn.Config.outputLayerNeurons
	LenbHiddenLayer2OutLayer := nn.Config.outputLayerNeurons
	sumtag := 0

	wInput2InputLayerData := AdjTmp[0:LenwInput2InputLayer]
	sumtag += LenwInput2InputLayer
	bInput2InputLayerData := AdjTmp[sumtag : LenbInput2InputLayer+sumtag]
	sumtag += LenbInput2InputLayer
	wInputLayer2HiddenLayerData := AdjTmp[sumtag : LenwInputLayer2HiddenLayer+sumtag]
	sumtag += LenwInputLayer2HiddenLayer
	bInputLayer2HiddenLayerData := AdjTmp[sumtag : LenbInputLayer2HiddenLayer+sumtag]
	sumtag += LenbInputLayer2HiddenLayer
	wHiddenLayer2OutLayerData := AdjTmp[sumtag : LenwHiddenLayer2OutLayer+sumtag]
	sumtag += LenwHiddenLayer2OutLayer
	bHiddenLayer2OutLayerData := AdjTmp[sumtag : LenbHiddenLayer2OutLayer+sumtag]

	wInput2InputLayerTmp := mat.NewDense(nn.Config.inputdataDims, nn.Config.inputLayerNeurons, wInput2InputLayerData)
	bInput2InputLayerTmp := mat.NewDense(1, nn.Config.inputLayerNeurons, bInput2InputLayerData)
	wInputLayer2HiddenLayerTmp := mat.NewDense(nn.Config.inputLayerNeurons, nn.Config.hiddenLayerNeurons, wInputLayer2HiddenLayerData)
	bInputLayer2HiddenLayerTmp := mat.NewDense(1, nn.Config.hiddenLayerNeurons, bInputLayer2HiddenLayerData)
	wHiddenLayer2OutLayerTmp := mat.NewDense(nn.Config.hiddenLayerNeurons, nn.Config.outputLayerNeurons, wHiddenLayer2OutLayerData)
	bHiddenLayer2OutLayerTmp := mat.NewDense(1, nn.Config.outputLayerNeurons, bHiddenLayer2OutLayerData)

	nn.wInput2InputLayerAdj.Add(nn.wInput2InputLayerAdj, wInput2InputLayerTmp)
	nn.bInput2InputLayerAdj.Add(nn.bInput2InputLayerAdj, bInput2InputLayerTmp)
	nn.wInputLayer2HiddenLayerAdj.Add(nn.wInputLayer2HiddenLayerAdj, wInputLayer2HiddenLayerTmp)
	nn.bInputLayer2HiddenLayerAdj.Add(nn.bInputLayer2HiddenLayerAdj, bInputLayer2HiddenLayerTmp)
	nn.wHiddenLayer2OutLayerAdj.Add(nn.wHiddenLayer2OutLayerAdj, wHiddenLayer2OutLayerTmp)
	nn.bHiddenLayer2OutLayerAdj.Add(nn.bHiddenLayer2OutLayerAdj, bHiddenLayer2OutLayerTmp)
	// fmt.Println("In Main Network, weights has been added from", rank, "with last element")
	// fmt.Println(nn.bHiddenLayer2OutLayerAdj.At(0, 0), bHiddenLayer2OutLayerTmp.At(0, 0), rank)
}

func (nn *NeuralNetFrame) PrepUpdatedWeightToTrainNet() []float64 {
	// fmt.Println(mat.Formatted(nn.bHiddenLayer2OutLayerAdj))
	wInput2InputLayerData := nn.wInput2InputLayerAdj.RawMatrix().Data
	bInput2InputLayerData := nn.bInput2InputLayerAdj.RawMatrix().Data
	wInputLayer2HiddenLayerData := nn.wInputLayer2HiddenLayerAdj.RawMatrix().Data
	bInputLayer2HiddenLayerData := nn.bInputLayer2HiddenLayerAdj.RawMatrix().Data
	wHiddenLayer2OutLayerData := nn.wHiddenLayer2OutLayerAdj.RawMatrix().Data
	bHiddenLayer2OutLayerData := nn.bHiddenLayer2OutLayerAdj.RawMatrix().Data

	LenwInput2InputLayer := nn.Config.inputdataDims * nn.Config.inputLayerNeurons
	LenbInput2InputLayer := nn.Config.inputLayerNeurons
	LenwInputLayer2HiddenLayer := nn.Config.inputLayerNeurons * nn.Config.hiddenLayerNeurons
	LenbInputLayer2HiddenLayer := nn.Config.hiddenLayerNeurons
	LenwHiddenLayer2OutLayer := nn.Config.hiddenLayerNeurons * nn.Config.outputLayerNeurons
	LenbHiddenLayer2OutLayer := nn.Config.outputLayerNeurons

	MPIDATA := make([]float64, LenwInput2InputLayer+LenbInput2InputLayer+LenwInputLayer2HiddenLayer+LenbInputLayer2HiddenLayer+LenwHiddenLayer2OutLayer+LenbHiddenLayer2OutLayer)
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
	return MPIDATA
}

func (nn *NeuralNetFrame) UpdatedWeightInTrainNet(MPIDATAdest []float64, i int, j int, rank int) {
	LenwInput2InputLayer := nn.Config.inputdataDims * nn.Config.inputLayerNeurons
	LenbInput2InputLayer := nn.Config.inputLayerNeurons
	LenwInputLayer2HiddenLayer := nn.Config.inputLayerNeurons * nn.Config.hiddenLayerNeurons
	LenbInputLayer2HiddenLayer := nn.Config.hiddenLayerNeurons
	LenwHiddenLayer2OutLayer := nn.Config.hiddenLayerNeurons * nn.Config.outputLayerNeurons
	LenbHiddenLayer2OutLayer := nn.Config.outputLayerNeurons
	sumtag := 0

	wInput2InputLayerData := MPIDATAdest[:LenwInput2InputLayer]
	sumtag += LenwInput2InputLayer
	bInput2InputLayerData := MPIDATAdest[sumtag : LenbInput2InputLayer+sumtag]
	sumtag += LenbInput2InputLayer
	wInputLayer2HiddenLayerData := MPIDATAdest[sumtag : LenwInputLayer2HiddenLayer+sumtag]
	sumtag += LenwInputLayer2HiddenLayer
	bInputLayer2HiddenLayerData := MPIDATAdest[sumtag : LenbInputLayer2HiddenLayer+sumtag]
	sumtag += LenbInputLayer2HiddenLayer
	wHiddenLayer2OutLayerData := MPIDATAdest[sumtag : LenwHiddenLayer2OutLayer+sumtag]
	sumtag += LenwHiddenLayer2OutLayer

	bHiddenLayer2OutLayerData := MPIDATAdest[sumtag : LenbHiddenLayer2OutLayer+sumtag]

	wInput2InputLayerDataAdj := mat.NewDense(nn.Config.inputdataDims, nn.Config.inputLayerNeurons, wInput2InputLayerData)
	bInput2InputLayerDataAdj := mat.NewDense(1, nn.Config.inputLayerNeurons, bInput2InputLayerData)
	wInputLayer2HiddenLayerDataAdj := mat.NewDense(nn.Config.inputLayerNeurons, nn.Config.hiddenLayerNeurons, wInputLayer2HiddenLayerData)
	bInputLayer2HiddenLayerDataAdj := mat.NewDense(1, nn.Config.hiddenLayerNeurons, bInputLayer2HiddenLayerData)
	wHiddenLayer2OutLayerDataAdj := mat.NewDense(nn.Config.hiddenLayerNeurons, nn.Config.outputLayerNeurons, wHiddenLayer2OutLayerData)
	bHiddenLayer2OutLayerDataAdj := mat.NewDense(1, nn.Config.outputLayerNeurons, bHiddenLayer2OutLayerData)

	nn.wInput2InputLayer.Add(nn.wInput2InputLayer, wInput2InputLayerDataAdj)
	nn.bInput2InputLayer.Add(nn.bInput2InputLayer, bInput2InputLayerDataAdj)
	nn.wInputLayer2HiddenLayer.Add(nn.wInputLayer2HiddenLayer, wInputLayer2HiddenLayerDataAdj)
	nn.bInputLayer2HiddenLayer.Add(nn.bInputLayer2HiddenLayer, bInputLayer2HiddenLayerDataAdj)
	nn.wHiddenLayer2OutLayer.Add(nn.wHiddenLayer2OutLayer, wHiddenLayer2OutLayerDataAdj)
	nn.bHiddenLayer2OutLayer.Add(nn.bHiddenLayer2OutLayer, bHiddenLayer2OutLayerDataAdj)
}
