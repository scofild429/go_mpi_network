package plotcode

import (
	"fmt"
	"os"
	"strconv"

	chart "github.com/wcharczuk/go-chart/v2"
)

func DrowLoss(Loss [][]float64, numEpochsenv int, parallelism int) {
	Xlabel := make([]float64, numEpochsenv)
	for i := range Xlabel {
		Xlabel[i] = float64(i + 1)
	}

	Charts := []chart.Series{}
	for i := 0; i < parallelism; i++ {
		charttmp := chart.ContinuousSeries{
			Name:    "The " + strconv.Itoa(i+1) + " Network",
			XValues: Xlabel,
			YValues: Loss[i],
		}
		Charts = append(Charts, charttmp)
	}
	graph := chart.Chart{
		XAxis: chart.XAxis{
			Name: "Training Epoches",
		},
		YAxis: chart.YAxis{
			Name: "Training Lossing",
		},
		Series: Charts,
	}

	graph.Elements = []chart.Renderable{
		chart.Legend(&graph),
	}

	f, _ := os.Create("Loss_output.png")
	defer f.Close()
	graph.Render(chart.PNG, f)
}

func DrowAccuracy(Accuracy [][]float64, numEpochsenv int, parallelism int) {
	Xlabel := make([]float64, numEpochsenv)
	for i := range Xlabel {
		Xlabel[i] = float64(i + 1)
	}

	Charts := []chart.Series{}
	for i := 0; i < parallelism; i++ {
		charttmp := chart.ContinuousSeries{
			Name:    "The " + strconv.Itoa(i+1) + " Network",
			XValues: Xlabel,
			YValues: Accuracy[i],
		}
		Charts = append(Charts, charttmp)
	}
	graph := chart.Chart{
		XAxis: chart.XAxis{
			Name: "Training Epoches",
		},
		YAxis: chart.YAxis{
			Name: "Training Accuracy",
		},
		Series: Charts,
	}

	graph.Elements = []chart.Renderable{
		chart.Legend(&graph),
	}

	f, _ := os.Create("Acc_output.png")
	defer f.Close()
	graph.Render(chart.PNG, f)
}

func DrowLossAllReduce(Loss []float64, numEpochsenv int) {
	Xlabel := make([]float64, numEpochsenv)
	for i := range Xlabel {
		Xlabel[i] = float64(i + 1)
	}

	Charts := []chart.Series{
		chart.ContinuousSeries{
			Name:    "The  Network",
			XValues: Xlabel,
			YValues: Loss,
		},
	}
	graph := chart.Chart{
		XAxis: chart.XAxis{
			Name: "Training Epoches",
		},
		YAxis: chart.YAxis{
			Name: "Training Lossing",
		},
		Series: Charts,
	}

	graph.Elements = []chart.Renderable{
		chart.Legend(&graph),
	}

	f, _ := os.Create("Loss_output_allreduce.png")
	defer f.Close()
	graph.Render(chart.PNG, f)
}

func DrowAccuracyAllReduce(Accuracy []float64, numEpochsenv int, rank int) {
	Xlabel := make([]float64, numEpochsenv)
	for i := range Xlabel {
		Xlabel[i] = float64(i + 1)
	}
	Charts := []chart.Series{
		chart.ContinuousSeries{
			Name:    "The  Network",
			XValues: Xlabel,
			YValues: Accuracy,
		},
	}
	graph := chart.Chart{
		XAxis: chart.XAxis{
			Name: "Training Epoches",
		},
		YAxis: chart.YAxis{
			Name: "Training Accuracy",
		},
		Series: Charts,
	}

	graph.Elements = []chart.Renderable{
		chart.Legend(&graph),
	}
	pngname := fmt.Sprintf("%d_Acc_output_allreduce.png", rank+1)
	f, _ := os.Create(pngname)
	defer f.Close()
	graph.Render(chart.PNG, f)
}
