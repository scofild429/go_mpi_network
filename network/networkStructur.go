package network

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type NeuralNete interface {
	Initialize()
	InitData()
	InitInputLayer()
	InitHiddenLayer()
	InitOutputLayer()
	FInputData2InputLayer(inputdata *mat.Dense, j int)
	NormalizationInputLayer()
	StandardizationInputLayer()

	SigmoidActiveFuncInputLayer()
	PrimeSigmoidActiveFuncInputLayer()
	RuleActiveFuncInputLayer()
	PrimeRuleActiveFuncOutputLayer()
	TanhActiveFuncInputLayer()
	PrimeTanhActiveFuncOutputLayer()

	FInputLayer2HiddenLayer(j int)
	NormalizationHiddenLayer()
	StandardizationHiddenLayer()

	SigmoidActiveFuncHiddenLayer()
	PrimeSigmoidActiveFuncHiddenLayer()
	RuleActiveFuncHiddenLayer()
	PrimeRuleActiveFuncHiddenLayer()
	TanhActiveFuncHiddenLayer()
	PrimeTanhActiveFuncHiddenLayer()

	FHiddenLayer2OutputLayer(j int)

	SigmoidActiveFuncOutputLayer()
	PrimeSigmoidActiveFuncOutputLayer()
	RuleActiveFuncOutputLayer()
	PrimeRuleActiveFuncInputLayer()
	TanhActiveFuncOutputLayer()
	PrimeTanhActiveFuncInputLayer()

	StandardizationOutputLayer()
	NormalizationOutputLayer()
	ApplySoftmaxAtEnd()
	Output(outputlabels *mat.Dense) float64
	Evaluation(outputlabels *mat.Dense) int64
	BOutput2OutputLayer(j int)
	BOutputLayer2HiddenLayer(j int)
	BHiddenLayer2InputLayer(inputdata *mat.Dense, j int)
	ReviewNeteWork()
	ReviewNeteWorkDims()
	PrepWeightToBoadcast(MPIDATA []float64)
	PrepUpdatedWeightToTrainNet() []float64
	ReciveInitialWeight(MPIDATA []float64, rank int)
	TrainWithEpochMPI(j int, trigger bool) float64
	NetworkForward(data *mat.Dense, j int)
	NetworkBackward(j int)
	NetworkEvaluation(j int) int
	PrepSendAdj(i int, j int, rank int) []float64
	UpdateWeightsMain(AdjTmp []float64, rank int)
	UpdatedWeightInTrainNet(MPIDATA []float64, i int, j int, rank int)
	ValidationEpoch(mark int, rank int) float64
	AccuWeightsMPI(j int)
	InitialWeightsMPISendRecv(j int)
	InitialWeightsMPIAllreduce(j int)
	UpdateWeightsSingleNode(j int)
}

type NeuralNetFrame struct {
	Config    NeuralNeteConfig
	inputdata []*mat.Dense
	testdata  []*mat.Dense
	validdata []*mat.Dense

	dInputLayerTmp       *mat.Dense
	dInputLayer          *mat.Dense
	wInput2InputLayer    *mat.Dense
	bInput2InputLayer    *mat.Dense
	wInput2InputLayerAdj *mat.Dense
	wInput2InputLayerMPI *mat.Dense
	bInput2InputLayerAdj *mat.Dense
	bInput2InputLayerMPI *mat.Dense
	dInputLayerErr       *mat.Dense
	dInputLayerSlope     *mat.Dense
	dInputLayerAdj       *mat.Dense

	dHiddenLayerTmp            *mat.Dense
	dHiddenLayer               *mat.Dense
	wInputLayer2HiddenLayer    *mat.Dense
	bInputLayer2HiddenLayer    *mat.Dense
	wInputLayer2HiddenLayerAdj *mat.Dense
	wInputLayer2HiddenLayerMPI *mat.Dense
	bInputLayer2HiddenLayerAdj *mat.Dense
	bInputLayer2HiddenLayerMPI *mat.Dense
	dHiddenLayerErr            *mat.Dense
	dHiddenLayerSlope          *mat.Dense
	dHiddenLayerAdj            *mat.Dense

	dOutputLayerTmp          *mat.Dense
	dOutputLayer             *mat.Dense
	wHiddenLayer2OutLayer    *mat.Dense
	bHiddenLayer2OutLayer    *mat.Dense
	wHiddenLayer2OutLayerAdj *mat.Dense
	wHiddenLayer2OutLayerMPI *mat.Dense
	bHiddenLayer2OutLayerAdj *mat.Dense
	bHiddenLayer2OutLayerMPI *mat.Dense
	dOutputLayerErr          *mat.Dense
	dOutputLayerLoss         *mat.Dense
	dOutputLayerSlope        *mat.Dense
	dOutputLayerAdj          *mat.Dense

	trainlabels []*mat.Dense
	testlabels  []*mat.Dense
	validlabels []*mat.Dense
}

type NeuralNeteConfig struct {
	traindataLen       int
	inputdataDims      int
	labelOnehotDims    int
	inputLayerNeurons  int
	hiddenLayerNeurons int
	outputLayerNeurons int
	outputDims         int
	learningRate       float64
	batchSize          int
	TrainbatchNum      int
	TestbatchNum       int
	ValidbatchNum      int
	testdataLen        int
	validdataLen       int
	randGen            *rand.Rand
	applyRule          func(int, int, float64) float64
	applyRulePrime     func(int, int, float64) float64
	applySigmode       func(int, int, float64) float64
	applySigmodePrime  func(int, int, float64) float64
	applySoftmax       func(int, int, float64) float64
	applyTanhPrime     func(int, int, float64) float64
	applyTanh          func(int, int, float64) float64
	applyL2loss        func(int, int, float64) float64
}
