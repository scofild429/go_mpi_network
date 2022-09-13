package singlenode

import (
	"log"
	"os"
	"strconv"

	"github.com/joho/godotenv"
	plotcode "github.com/scofild429/goai/goai/plot"
	"github.com/scofild429/goai/myData"
	"github.com/scofild429/goai/network"
)

func Single_node_iris(nompi bool) {
	//read .env configuation
	err := godotenv.Load(".irisenv")
	if err != nil {
		log.Fatalf("Some error occured when load configuation file . Err: %s", err)
	}
	numEpochsenv, _ := strconv.Atoi(os.Getenv("numEpochs"))

	parallelism := 1
	rank := 1
	myData.LoadDataIris(0.8, 0.1, 0.1, parallelism, rank)

	net, networker := network.Initialization()

	var Loss []float64
	var Accuracy []float64
	for i := 0; i < numEpochsenv; i++ {
		losstmp := 0.0
		for j := 0; j < net.Config.TrainbatchNum; j++ {
			losstmp += networker.TrainWithEpochMPI(j, nompi)
		}
		Loss = append(Loss, losstmp)
		accuracytmp := networker.ValidationEpoch(i, rank)
		Accuracy = append(Accuracy, accuracytmp)
	}
	LossAll := make([][]float64, parallelism)
	AccuracyAll := make([][]float64, parallelism)
	LossAll[0] = Loss
	AccuracyAll[0] = Accuracy
	plotcode.DrowLoss(LossAll, numEpochsenv, parallelism)
	plotcode.DrowAccuracy(AccuracyAll, numEpochsenv, parallelism)
}
