package mpicode

import (
	"log"
	"os"
	"strconv"

	"github.com/joho/godotenv"
	mpi "github.com/sbromberger/gompi"
	plotcode "github.com/scofild429/goai/goai/plot"
	"github.com/scofild429/goai/myData"
	"github.com/scofild429/goai/network"
)

func Mpi_iris_Allreduce() {
	//read .env configuation
	err := godotenv.Load(".irisenv")
	if err != nil {
		log.Fatalf("Some error occured when load configuation file . Err: %s", err)
	}
	inputdataDimsenv, _ := strconv.Atoi(os.Getenv("inputdataDims"))
	inputLayerNeuronsenv, _ := strconv.Atoi(os.Getenv("inputLayerNeurons"))
	hiddenLayerNeuronsenv, _ := strconv.Atoi(os.Getenv("hiddenLayerNeurons"))
	outputLayerNeuronsenv, _ := strconv.Atoi(os.Getenv("outputLayerNeurons"))
	numEpochsenv, _ := strconv.Atoi(os.Getenv("numEpochs"))
	MPIDATALEN := inputdataDimsenv*inputLayerNeuronsenv + inputLayerNeuronsenv + inputLayerNeuronsenv*hiddenLayerNeuronsenv + hiddenLayerNeuronsenv + hiddenLayerNeuronsenv*outputLayerNeuronsenv + outputLayerNeuronsenv

	mpi.Start(true)
	var ranks []int
	newComm := mpi.NewCommunicator(ranks)

	MPIDATA := make([]float64, MPIDATALEN)
	MPIDATAdest := make([]float64, MPIDATALEN)
	parallelism := mpi.WorldSize()
	rank := mpi.WorldRank()

	myData.LoadDataIris(0.8, 0.1, 0.1, parallelism, rank)

	net, networker := network.Initialization()

	if newComm.Rank() == 0 {
		networker.PrepWeightToBoadcast(MPIDATA)
	}
	newComm.BcastFloat64s(MPIDATA, 0)
	networker.ReciveInitialWeight(MPIDATA, rank)

	var Loss []float64
	var Accuracy []float64
	for i := 0; i < numEpochsenv; i++ {
		losstmp := 0.0
		for j := 0; j < net.Config.TrainbatchNum; j++ {
			// fmt.Println("Starting for", i, "-th epoches", j, "-th batch ", rank, "-th train netework start  training")
			losstmp += networker.TrainWithEpochMPI(j, false)
			MPIDATA = networker.PrepSendAdj(i, j, rank)
			newComm.AllreduceFloat64s(MPIDATAdest, MPIDATA, mpi.OpSum, 0)
			networker.UpdatedWeightInTrainNet(MPIDATAdest, i, j, rank)
		}
		Loss = append(Loss, losstmp)
		accuracytmp := networker.ValidationEpoch(i, rank)
		Accuracy = append(Accuracy, accuracytmp)
	}
	if newComm.Rank() != 0 {
		newComm.SendFloat64s(Loss, 0, 0)
		newComm.SendFloat64s(Accuracy, 0, 1)
	}

	LossAll := make([][]float64, parallelism)
	AccuracyAll := make([][]float64, parallelism)
	for i := range LossAll {
		LossAll[i] = make([]float64, numEpochsenv)
		AccuracyAll[i] = make([]float64, numEpochsenv)
	}

	if newComm.Rank() == 0 {
		LossAll[0] = Loss
		AccuracyAll[0] = Accuracy
		for mark := 1; mark < parallelism; mark++ {
			LossAll[mark], _ = newComm.RecvFloat64s(mark, 0)
			AccuracyAll[mark], _ = newComm.RecvFloat64s(mark, 1)
		}
		plotcode.DrowLoss(LossAll, numEpochsenv, parallelism)
		plotcode.DrowAccuracy(AccuracyAll, numEpochsenv, parallelism)
	}
	mpi.Stop()
}

func Mpi_images_Allreduce() {
	//read .env configuation
	err := godotenv.Load(".imgenv")
	if err != nil {
		log.Fatalf("Some error occured when load configuation file . Err: %s", err)
	}
	inputdataDimsenv, _ := strconv.Atoi(os.Getenv("inputdataDims"))
	inputLayerNeuronsenv, _ := strconv.Atoi(os.Getenv("inputLayerNeurons"))
	hiddenLayerNeuronsenv, _ := strconv.Atoi(os.Getenv("hiddenLayerNeurons"))
	outputLayerNeuronsenv, _ := strconv.Atoi(os.Getenv("outputLayerNeurons"))
	numEpochsenv, _ := strconv.Atoi(os.Getenv("numEpochs"))
	MPIDATALEN := inputdataDimsenv*inputLayerNeuronsenv + inputLayerNeuronsenv + inputLayerNeuronsenv*hiddenLayerNeuronsenv + hiddenLayerNeuronsenv + hiddenLayerNeuronsenv*outputLayerNeuronsenv + outputLayerNeuronsenv

	mpi.Start(true)
	var ranks []int
	newComm := mpi.NewCommunicator(ranks)

	MPIDATA := make([]float64, MPIDATALEN)
	MPIDATAdest := make([]float64, MPIDATALEN)
	parallelism := mpi.WorldSize()
	rank := mpi.WorldRank()

	myData.ReadImage(parallelism, rank, 0.9)

	net, networker := network.Initialization()

	if newComm.Rank() == 0 {
		networker.PrepWeightToBoadcast(MPIDATA)
	}
	newComm.BcastFloat64s(MPIDATA, 0)
	networker.ReciveInitialWeight(MPIDATA, rank)

	var Loss []float64
	var Accuracy []float64
	for i := 0; i < numEpochsenv; i++ {
		losstmp := 0.0
		for j := 0; j < net.Config.TrainbatchNum; j++ {
			// fmt.Println("Starting for", i, "-th epoches", j, "-th batch", rank, "-th train netework start  training")
			losstmp += networker.TrainWithEpochMPI(j, false)
			MPIDATA = networker.PrepSendAdj(i, j, rank)
			newComm.AllreduceFloat64s(MPIDATAdest, MPIDATA, mpi.OpSum, 0)
			networker.UpdatedWeightInTrainNet(MPIDATAdest, i, j, rank)
		}
		Loss = append(Loss, losstmp)
		accuracytmp := networker.ValidationEpoch(i, rank)
		Accuracy = append(Accuracy, accuracytmp)
	}
	if newComm.Rank() != 0 {
		newComm.SendFloat64s(Loss, 0, 0)
		newComm.SendFloat64s(Accuracy, 0, 1)
	}
	LossAll := make([][]float64, parallelism)
	AccuracyAll := make([][]float64, parallelism)
	for i := range LossAll {
		LossAll[i] = make([]float64, numEpochsenv)
		AccuracyAll[i] = make([]float64, numEpochsenv)
	}
	if newComm.Rank() == 0 {
		LossAll[0] = Loss
		AccuracyAll[0] = Accuracy
		for mark := 1; mark < parallelism; mark++ {
			LossAll[mark], _ = newComm.RecvFloat64s(mark, 0)
			AccuracyAll[mark], _ = newComm.RecvFloat64s(mark, 1)
		}
		// fmt.Println(AccuracyAll)
		plotcode.DrowLoss(LossAll, numEpochsenv, parallelism)
		plotcode.DrowAccuracy(AccuracyAll, numEpochsenv, parallelism)
	}
	mpi.Stop()
}
