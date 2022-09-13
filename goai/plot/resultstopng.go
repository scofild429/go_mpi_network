package plotcode

import (
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
		Background: chart.Style{
			Padding: chart.Box{
				Top:  10,
				Left: 140,
			},
		},
		XAxis: chart.XAxis{
			Name: "Training Epoches",
		},
		YAxis: chart.YAxis{
			Name: "Training Lossing",
		},
		Series: Charts,
	}

	graph.Elements = []chart.Renderable{
		chart.LegendLeft(&graph),
	}
	graph.Height = 600
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
		Background: chart.Style{
			Padding: chart.Box{
				Top:  10,
				Left: 140,
			},
		},
		XAxis: chart.XAxis{
			Name: "Training Epoches",
		},
		YAxis: chart.YAxis{
			Name: "Training Accuracy",
		},
		Series: Charts,
	}

	graph.Elements = []chart.Renderable{
		chart.LegendLeft(&graph),
	}
	graph.Height = 600
	f, _ := os.Create("Acc_output.png")
	defer f.Close()
	graph.Render(chart.PNG, f)
}
