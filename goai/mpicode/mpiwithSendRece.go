package mpicode

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/joho/godotenv"
	mpi "github.com/sbromberger/gompi"
	plotcode "github.com/scofild429/goai/goai/plot"
	"github.com/scofild429/goai/myData"
	"github.com/scofild429/goai/network"
)

func Mpi_iris_SendRecv() {
	start := time.Now()

	// start mpi
	mpi.Start(true)
	var ranks []int
	newComm := mpi.NewCommunicator(ranks)
	parallelism := mpi.WorldSize()
	rank := mpi.WorldRank()

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
	MPIDATA := make([]float64, MPIDATALEN)

	myData.LoadDataIris(0.8, 0.1, 0.1, parallelism-1, rank)
	///////////////////////////////////////////////////////////////////////////////////////
	////  mpirun -n -4 ./goai
	////  There is a problem remains, even we just three train networks, but its train dataset
	////  has to be divied into 4 part, because MPI send recv error occurs when we only divie
	////  them into 3 parts
	////
	//////////////////////////////////////////////////////////////////////////////////////

	net, networker := network.Initialization()
	if rank == 0 {
		networker.PrepWeightToBoadcast(MPIDATA)
	}
	newComm.BcastFloat64s(MPIDATA, 0)

	if rank != 0 {
		networker.ReciveInitialWeight(MPIDATA, rank)
		for i := 0; i < numEpochsenv; i++ {
			loss := 0.0
			for j := 0; j < net.Config.TrainbatchNum; j++ {
				// fmt.Println("Starting for", i, "-th epoches", j, "-th batch  in node ", rank, "for training")
				loss += networker.TrainWithEpochMPI(j, false)
				MPIDATA = networker.PrepSendAdj(i, j, rank)
				newComm.SendFloat64s(MPIDATA, 0, rank)
				MPIDATA, _ = newComm.RecvFloat64s(0, rank)
				networker.UpdatedWeightInTrainNet(MPIDATA, i, j, rank)
			}
			newComm.SendFloat64(loss, 0, rank)
			accuracy := networker.ValidationEpoch(i, rank)
			newComm.SendFloat64(accuracy, 0, rank+1)
		}
	}
	if rank == 0 {
		Loss := make([][]float64, parallelism-1)
		Accuracy := make([][]float64, parallelism-1)
		for i := range Loss {
			Loss[i] = make([]float64, numEpochsenv)
			Accuracy[i] = make([]float64, numEpochsenv)
		}
		for i := 0; i < numEpochsenv; i++ {
			for j := 0; j < net.Config.TrainbatchNum; j++ {
				// fmt.Println("Starting for", i, "-th epoches", j, "-th batch  in Main netework weights to be merged")
				networker.InitialWeightsMPIAllreduce(j)
				for mark := 1; mark < parallelism; mark++ {
					AdjTmp, _ := newComm.RecvFloat64s(mark, mark)
					networker.UpdateWeightsMain(AdjTmp, mark)
				}
				MPIDATA = networker.PrepUpdatedWeightToTrainNet()
				for mark := 1; mark < parallelism; mark++ {
					newComm.SendFloat64s(MPIDATA, mark, mark)
				}
			}
			for mark := 1; mark < parallelism; mark++ {
				Loss[mark-1][i], _ = newComm.RecvFloat64(mark, mark)
				Accuracy[mark-1][i], _ = newComm.RecvFloat64(mark, mark+1)
			}
		}
		plotcode.DrowLoss(Loss, numEpochsenv, parallelism-1)
		plotcode.DrowAccuracy(Accuracy, numEpochsenv, parallelism-1)
	}
	// cal time consuming
	end := time.Now()
	timeconsumeeeachnode := make([]float64, 0)
	timeconsumeeeachnode = append(timeconsumeeeachnode, end.Sub(start).Seconds())
	timeconsumeaverage := make([]float64, 1)
	newComm.AllreduceFloat64s(timeconsumeaverage, timeconsumeeeachnode, mpi.OpSum, 0)
	if newComm.Rank() == 0 {
		fmt.Println(timeconsumeaverage[0] / float64(parallelism))
	}
	mpi.Stop()
}

func Mpi_images_SendRecv() {
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
	MPIDATA := make([]float64, MPIDATALEN)

	mpi.Start(true)
	var ranks []int
	newComm := mpi.NewCommunicator(ranks)
	parallelism := mpi.WorldSize()
	rank := mpi.WorldRank()

	myData.ReadImage(parallelism, rank, 0.9)
	net := new(network.NeuralNetFrame)

	net, networker := network.Initialization()
	if newComm.Rank() == 0 {
		networker.PrepWeightToBoadcast(MPIDATA)
	}
	newComm.BcastFloat64s(MPIDATA, 0)

	if rank != 0 {
		networker.ReciveInitialWeight(MPIDATA, rank)
		for i := 0; i < numEpochsenv; i++ {
			loss := 0.0
			for j := 0; j < net.Config.TrainbatchNum; j++ {
				// fmt.Println("Starting for", i, "-th epoches", j, "-th batch  in node ", rank, "for training")
				loss += networker.TrainWithEpochMPI(j, false)
				MPIDATA = networker.PrepSendAdj(i, j, rank)
				newComm.SendFloat64s(MPIDATA, 0, rank)
				MPIDATA, _ = newComm.RecvFloat64s(0, rank)
				networker.UpdatedWeightInTrainNet(MPIDATA, i, j, rank)
			}
			newComm.SendFloat64(loss, 0, rank)
			accuracy := networker.ValidationEpoch(i, rank)
			newComm.SendFloat64(accuracy, 0, rank+1)
		}
	}
	if newComm.Rank() == 0 {
		Loss := make([][]float64, parallelism-1)
		Accuracy := make([][]float64, parallelism)
		for i := range Loss {
			Loss[i] = make([]float64, numEpochsenv)
			Accuracy[i] = make([]float64, numEpochsenv)
		}
		for i := 0; i < numEpochsenv; i++ {
			for j := 0; j < net.Config.TrainbatchNum; j++ {
				// fmt.Println("Starting for", i, "-th epoches", j, "-th batch  in Main network weights  has to be merged")
				networker.InitialWeightsMPIAllreduce(j)
				for mark := 1; mark < parallelism; mark++ {
					AdjTmp, _ := newComm.RecvFloat64s(mark, mark)
					networker.UpdateWeightsMain(AdjTmp, mark)
				}
				MPIDATA = networker.PrepUpdatedWeightToTrainNet()
				for mark := 1; mark < parallelism; mark++ {
					newComm.SendFloat64s(MPIDATA, mark, mark)
				}
			}
			for mark := 1; mark < parallelism; mark++ {
				Loss[mark-1][i], _ = newComm.RecvFloat64(mark, mark)
				Accuracy[mark-1][i], _ = newComm.RecvFloat64(mark, mark+1)
			}
		}
		plotcode.DrowLoss(Loss, numEpochsenv, parallelism-1)
		plotcode.DrowAccuracy(Accuracy, numEpochsenv, parallelism-1)
	}
	mpi.Stop()
}
