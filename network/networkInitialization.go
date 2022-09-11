package network

import (
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/scofild429/goai/myData"
	"gonum.org/v1/gonum/mat"
)

func (nn *NeuralNetFrame) Initialize() {
	batchSizeenv, _ := strconv.Atoi(os.Getenv("batchSize"))
	inputLayerNeuronsenv, _ := strconv.Atoi(os.Getenv("inputLayerNeurons"))
	hiddenLayerNeuronsenv, _ := strconv.Atoi(os.Getenv("hiddenLayerNeurons"))
	outputLayerNeuronsenv, _ := strconv.Atoi(os.Getenv("outputLayerNeurons"))
	learningRateenv, _ := strconv.ParseFloat(os.Getenv("learningRate"), 64)
	inputdataDimsenv, _ := strconv.Atoi(os.Getenv("inputdataDims"))
	labelOnehotDimsenv, _ := strconv.Atoi(os.Getenv("labelOnehotDims"))

	nn.Config = NeuralNeteConfig{
		inputLayerNeurons:  inputLayerNeuronsenv,
		hiddenLayerNeurons: hiddenLayerNeuronsenv,
		outputLayerNeurons: outputLayerNeuronsenv,
		learningRate:       learningRateenv,
		batchSize:          batchSizeenv,
		inputdataDims:      inputdataDimsenv,
		labelOnehotDims:    labelOnehotDimsenv,
	}

	randSource := rand.NewSource(time.Now().Unix())
	nn.Config.randGen = rand.New(randSource)
	nn.Config.applySigmode = func(r, c int, v float64) float64 { return sigmoid(v) }
	nn.Config.applySigmodePrime = func(r, c int, v float64) float64 { return sigmoidPrime(v) }
	nn.Config.applyL2loss = func(_, _ int, v float64) float64 { return v * v / 2.0 }
	nn.Config.applyRule = func(r, c int, v float64) float64 { return rule(v) }
	nn.Config.applyRulePrime = func(r, c int, v float64) float64 { return rulePrime(v) }
	nn.Config.applyTanh = func(r, c int, v float64) float64 { return tanh(v) }
	nn.Config.applyTanhPrime = func(r, c int, v float64) float64 { return tanhPrime(v) }

}

func (nn *NeuralNetFrame) InitDataForTest() {
	trainDataLen := len(myData.TrainingData)
	nn.Config.traindataLen = trainDataLen
	nn.Config.TrainbatchNum = trainDataLen / nn.Config.batchSize
	for i := 0; i < nn.Config.TrainbatchNum; i++ {
		s0 := mat.NewDense(1, 4, []float64{1, 0, 0, 0})
		s1 := mat.NewDense(1, 4, []float64{0, 1, 0, 0})
		s2 := mat.NewDense(1, 4, []float64{0, 1, 0, 1})
		s3 := mat.NewDense(1, 4, []float64{1, 1, 0, 1})
		nn.inputdata = append(nn.inputdata, s0)
		nn.inputdata = append(nn.inputdata, s1)
		nn.inputdata = append(nn.inputdata, s2)
		nn.inputdata = append(nn.inputdata, s3)

		t0 := mat.NewDense(1, 3, []float64{1, 0, 0})
		t1 := mat.NewDense(1, 3, []float64{1, 0, 0})
		t2 := mat.NewDense(1, 3, []float64{0, 0, 1})
		t3 := mat.NewDense(1, 3, []float64{1, 0, 1})
		nn.trainlabels = append(nn.trainlabels, t0)
		nn.trainlabels = append(nn.trainlabels, t1)
		nn.trainlabels = append(nn.trainlabels, t2)
		nn.trainlabels = append(nn.trainlabels, t3)
	}
	testDataLen := len(myData.TestingData)
	nn.Config.testdataLen = testDataLen
	nn.Config.TestbatchNum = testDataLen / nn.Config.batchSize
	for i := 0; i < testDataLen; i++ {
		nn.testdata = append(nn.testdata, &myData.TestingData[i])
		nn.trainlabels = append(nn.trainlabels, &myData.TestingTarget[i])
	}
	validDataLen := len(myData.ValidingData)
	nn.Config.validdataLen = validDataLen
	nn.Config.ValidbatchNum = validDataLen / nn.Config.batchSize
	for i := 0; i < validDataLen; i++ {
		nn.validdata = append(nn.validdata, &myData.ValidingData[i])
		nn.validlabels = append(nn.validlabels, &myData.ValidingTarget[i])
	}
}

func (nn *NeuralNetFrame) InitData() {
	trainDataLen := len(myData.TrainingData)
	nn.Config.traindataLen = trainDataLen
	nn.Config.TrainbatchNum = trainDataLen / nn.Config.batchSize
	for i := 0; i < trainDataLen; i++ {
		nn.inputdata = append(nn.inputdata, &myData.TrainingData[i])
		nn.trainlabels = append(nn.trainlabels, &myData.TrainingTarget[i])
	}
	testDataLen := len(myData.TestingData)
	nn.Config.testdataLen = testDataLen
	nn.Config.TestbatchNum = testDataLen / nn.Config.batchSize
	for i := 0; i < testDataLen; i++ {
		nn.testdata = append(nn.testdata, &myData.TestingData[i])
		nn.trainlabels = append(nn.trainlabels, &myData.TestingTarget[i])
	}
	validDataLen := len(myData.ValidingData)
	nn.Config.validdataLen = validDataLen
	nn.Config.ValidbatchNum = validDataLen / nn.Config.batchSize
	for i := 0; i < validDataLen; i++ {
		nn.validdata = append(nn.validdata, &myData.ValidingData[i])
		nn.validlabels = append(nn.validlabels, &myData.ValidingTarget[i])
	}
}

func (nn *NeuralNetFrame) InitDataWithBatchsize() {
	//      train data
	trainDataLen := int(math.Floor(float64(len(myData.TrainingData))/float64(nn.Config.batchSize)) * float64(nn.Config.batchSize))
	nn.Config.traindataLen = trainDataLen
	for i := 0; i < trainDataLen; i += nn.Config.batchSize {
		if i+3 < len(myData.TrainingData) {
			s0 := myData.TrainingData[i].RawRowView(0)
			s1 := myData.TrainingData[i+1].RawRowView(0)
			s2 := myData.TrainingData[i+2].RawRowView(0)
			s3 := myData.TrainingData[i+3].RawRowView(0)
			s0 = append(s0, s1...)
			s0 = append(s0, s2...)
			s0 = append(s0, s3...)
			trainingdatabatch := mat.NewDense(1, nn.Config.inputdataDims*4, s0)
			nn.inputdata = append(nn.inputdata, trainingdatabatch)
		}
	}

	for i := 0; i < trainDataLen; i += nn.Config.batchSize {
		if i+3 < len(myData.TrainingTarget) {
			s0 := myData.TrainingTarget[i].RawRowView(0)
			s1 := myData.TrainingTarget[i+1].RawRowView(0)
			s2 := myData.TrainingTarget[i+2].RawRowView(0)
			s3 := myData.TrainingTarget[i+3].RawRowView(0)
			s0 = append(s0, s1...)
			s0 = append(s0, s2...)
			s0 = append(s0, s3...)
			traingtargetbatch := mat.NewDense(1, nn.Config.labelOnehotDims*4, s0)
			nn.trainlabels = append(nn.trainlabels, traingtargetbatch)
		}
	}
	nn.Config.TrainbatchNum = trainDataLen / nn.Config.batchSize
	//      test data
	testDataLen := int(math.Floor(float64(len(myData.TestingData))/float64(nn.Config.batchSize)) * float64(nn.Config.batchSize))
	nn.Config.testdataLen = testDataLen
	for i := 0; i < testDataLen; i += nn.Config.batchSize {
		if i+3 < len(myData.TestingData) {
			s0 := myData.TestingData[i].RawRowView(0)
			s1 := myData.TestingData[i+1].RawRowView(0)
			s2 := myData.TestingData[i+2].RawRowView(0)
			s3 := myData.TestingData[i+3].RawRowView(0)
			s0 = append(s0, s1...)
			s0 = append(s0, s2...)
			s0 = append(s0, s3...)
			testingdatabatch := mat.NewDense(1, nn.Config.inputdataDims*4, s0)
			nn.testdata = append(nn.testdata, testingdatabatch)
		}
	}

	for i := 0; i < testDataLen; i += nn.Config.batchSize {
		if i+3 < len(myData.TestingTarget) {
			s0 := myData.TestingTarget[i].RawRowView(0)
			s1 := myData.TestingTarget[i+1].RawRowView(0)
			s2 := myData.TestingTarget[i+2].RawRowView(0)
			s3 := myData.TestingTarget[i+3].RawRowView(0)
			s0 = append(s0, s1...)
			s0 = append(s0, s2...)
			s0 = append(s0, s3...)
			testingtargetbatch := mat.NewDense(1, nn.Config.labelOnehotDims*4, s0)
			nn.testlabels = append(nn.testlabels, testingtargetbatch)
		}
	}
	nn.Config.TestbatchNum = testDataLen / nn.Config.batchSize

	// valid data
	validDataLen := int(math.Floor(float64(len(myData.ValidingData))/float64(nn.Config.batchSize)) * float64(nn.Config.batchSize))
	nn.Config.validdataLen = validDataLen
	for i := 0; i < validDataLen; i += nn.Config.batchSize {
		if i+3 < len(myData.ValidingData) {
			s0 := myData.ValidingData[i].RawRowView(0)
			s1 := myData.ValidingData[i+1].RawRowView(0)
			s2 := myData.ValidingData[i+2].RawRowView(0)
			s3 := myData.ValidingData[i+3].RawRowView(0)
			s0 = append(s0, s1...)
			s0 = append(s0, s2...)
			s0 = append(s0, s3...)
			validingdatabatch := mat.NewDense(1, nn.Config.inputdataDims*4, s0)
			nn.validdata = append(nn.validdata, validingdatabatch)
		}
	}

	for i := 0; i < validDataLen; i += nn.Config.batchSize {
		if i+3 < len(myData.ValidingTarget) {
			s0 := myData.ValidingTarget[i].RawRowView(0)
			s1 := myData.ValidingTarget[i+1].RawRowView(0)
			s2 := myData.ValidingTarget[i+2].RawRowView(0)
			s3 := myData.ValidingTarget[i+3].RawRowView(0)
			s0 = append(s0, s1...)
			s0 = append(s0, s2...)
			s0 = append(s0, s3...)
			validingtargetbatch := mat.NewDense(1, nn.Config.labelOnehotDims*4, s0)
			nn.validlabels = append(nn.validlabels, validingtargetbatch)
		}
	}
	nn.Config.ValidbatchNum = validDataLen / nn.Config.batchSize
}

func (nn *NeuralNetFrame) InitInputLayer() {
	wInputRaw2InputLayer := make([]float64, nn.Config.inputdataDims*nn.Config.inputLayerNeurons)
	bInputRaw2InputLayer := make([]float64, nn.Config.inputLayerNeurons)
	for _, param := range [][]float64{wInputRaw2InputLayer, bInputRaw2InputLayer} {
		for i := range param {
			param[i] = nn.Config.randGen.Float64()
			// param[i] = 1
		}
	}
	nn.wInput2InputLayer = mat.NewDense(nn.Config.inputdataDims, nn.Config.inputLayerNeurons, wInputRaw2InputLayer)
	nn.bInput2InputLayer = mat.NewDense(1, nn.Config.inputLayerNeurons, bInputRaw2InputLayer)

	nn.dInputLayer = &mat.Dense{}
	nn.dInputLayerTmp = &mat.Dense{}
	nn.dInputLayerErr = &mat.Dense{}
	nn.dInputLayerSlope = &mat.Dense{}
	nn.wInput2InputLayerAdj = mat.NewDense(nn.Config.inputdataDims, nn.Config.inputLayerNeurons, nil)
	nn.bInput2InputLayerAdj = mat.NewDense(1, nn.Config.inputLayerNeurons, nil)
	nn.wInput2InputLayerMPI = mat.NewDense(nn.Config.inputdataDims, nn.Config.inputLayerNeurons, nil)
	nn.bInput2InputLayerMPI = mat.NewDense(1, nn.Config.inputLayerNeurons, nil)
	nn.dInputLayerAdj = &mat.Dense{}
}

func (nn *NeuralNetFrame) InitHiddenLayer() {
	wInputLayer2HiddenRaw := make([]float64, nn.Config.inputLayerNeurons*nn.Config.hiddenLayerNeurons)
	bInputLayer2HiddenRaw := make([]float64, nn.Config.hiddenLayerNeurons)
	for _, param := range [][]float64{wInputLayer2HiddenRaw, bInputLayer2HiddenRaw} {
		for i := range param {
			param[i] = nn.Config.randGen.Float64()
			// param[i] = 1
		}
	}
	nn.wInputLayer2HiddenLayer = mat.NewDense(nn.Config.inputLayerNeurons, nn.Config.hiddenLayerNeurons, wInputLayer2HiddenRaw)
	nn.bInputLayer2HiddenLayer = mat.NewDense(1, nn.Config.hiddenLayerNeurons, bInputLayer2HiddenRaw)
	nn.dHiddenLayer = &mat.Dense{}
	nn.dHiddenLayerTmp = &mat.Dense{}
	nn.dHiddenLayerErr = &mat.Dense{}
	nn.dHiddenLayerSlope = &mat.Dense{}
	nn.wInputLayer2HiddenLayerAdj = mat.NewDense(nn.Config.inputLayerNeurons, nn.Config.hiddenLayerNeurons, nil)
	nn.bInputLayer2HiddenLayerAdj = mat.NewDense(1, nn.Config.hiddenLayerNeurons, nil)
	nn.wInputLayer2HiddenLayerMPI = mat.NewDense(nn.Config.inputLayerNeurons, nn.Config.hiddenLayerNeurons, nil)
	nn.bInputLayer2HiddenLayerMPI = mat.NewDense(1, nn.Config.hiddenLayerNeurons, nil)
	nn.dHiddenLayerAdj = &mat.Dense{}
}

func (nn *NeuralNetFrame) InitOutputLayer() {
	wHiddenLayer2OutRaw := make([]float64, nn.Config.hiddenLayerNeurons*nn.Config.outputLayerNeurons)
	bHiddenLayer2OutRaw := make([]float64, nn.Config.outputLayerNeurons)

	for _, param := range [][]float64{wHiddenLayer2OutRaw, bHiddenLayer2OutRaw} {
		for i := range param {
			param[i] = nn.Config.randGen.Float64()
			// param[i] = 1
		}
	}
	nn.wHiddenLayer2OutLayer = mat.NewDense(nn.Config.hiddenLayerNeurons, nn.Config.outputLayerNeurons, wHiddenLayer2OutRaw)
	nn.bHiddenLayer2OutLayer = mat.NewDense(1, nn.Config.outputLayerNeurons, bHiddenLayer2OutRaw)
	nn.dOutputLayer = mat.NewDense(1, nn.Config.outputLayerNeurons, nil)
	nn.dOutputLayerTmp = &mat.Dense{}
	nn.dOutputLayerErr = &mat.Dense{}
	nn.dOutputLayerLoss = &mat.Dense{}
	nn.dOutputLayerSlope = &mat.Dense{}
	nn.wHiddenLayer2OutLayerAdj = mat.NewDense(nn.Config.hiddenLayerNeurons, nn.Config.outputLayerNeurons, nil)
	nn.bHiddenLayer2OutLayerAdj = mat.NewDense(1, nn.Config.outputLayerNeurons, nil)
	nn.wHiddenLayer2OutLayerMPI = mat.NewDense(nn.Config.hiddenLayerNeurons, nn.Config.outputLayerNeurons, nil)
	nn.bHiddenLayer2OutLayerMPI = mat.NewDense(1, nn.Config.outputLayerNeurons, nil)
	nn.dOutputLayerAdj = &mat.Dense{}
}
