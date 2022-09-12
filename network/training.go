package network

import (
	"gonum.org/v1/gonum/mat"
)

func Initialization() (*NeuralNetFrame, NeuralNete) {
	net := new(NeuralNetFrame)
	var networker NeuralNete
	networker = net
	networker.Initialize()
	networker.InitData()

	networker.InitInputLayer()
	networker.InitHiddenLayer()
	networker.InitOutputLayer()

	return net, networker
}
func (nn *NeuralNetFrame) TrainWithEpochMPI(j int, nompi bool) float64 {
	trainloss := 0.0
	nn.InitialWeightsMPISendRecv(j)
	for i := 0; i < nn.Config.batchSize; i++ {
		nn.NetworkForward(nn.inputdata[j*nn.Config.batchSize+i], j*nn.Config.batchSize+i)
		trainloss += nn.Output(nn.trainlabels[j*nn.Config.batchSize+i])
		nn.NetworkBackward(j*nn.Config.batchSize + i)
		nn.AccuWeightsMPI(j*nn.Config.batchSize + i)
		if nompi {
			nn.UpdateWeightsSingleNode(j*nn.Config.batchSize + i)
		}

	}
	return trainloss / float64(nn.Config.testdataLen)
}

func (nn *NeuralNetFrame) NetworkForward(data *mat.Dense, j int) {
	nn.FInputData2InputLayer(data, j)
	nn.StandardizationInputLayer()
	nn.SigmoidActiveFuncInputLayer()

	nn.FInputLayer2HiddenLayer(j)
	nn.StandardizationHiddenLayer()
	nn.SigmoidActiveFuncHiddenLayer()

	nn.FHiddenLayer2OutputLayer(j)
	// nn.StandardizationOutputLayer()
	nn.SigmoidActiveFuncOutputLayer()
}

func (nn *NeuralNetFrame) NetworkBackward(j int) {
	nn.PrimeSigmoidActiveFuncOutputLayer()
	nn.BOutput2OutputLayer(j)

	nn.PrimeSigmoidActiveFuncHiddenLayer()
	nn.BOutputLayer2HiddenLayer(j)

	nn.PrimeSigmoidActiveFuncInputLayer()
	nn.BHiddenLayer2InputLayer(nn.inputdata[j], j)

}

func (nn *NeuralNetFrame) ValidationEpoch(epoch int, rank int) float64 {
	correctNum := 0
	for j := 0; j < nn.Config.validdataLen; j++ {
		correctNum += nn.NetworkEvaluation(j, rank)
	}
	AccuracyTemp := float64(correctNum) / float64(nn.Config.validdataLen)
	return float64(AccuracyTemp)
}

func (nn *NeuralNetFrame) NetworkEvaluation(j int, rank int) int {
	nn.NetworkForward(nn.validdata[j], j)
	count := nn.Evaluation(nn.validlabels[j], j, rank)
	return int(count)
}
