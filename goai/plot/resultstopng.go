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

func PlotIrisSpeedup() {
	Charts := []chart.Series{}

	irisAllreducetime := []float64{1.4, 0.91, 0.9, 0.84, 0.86, 0.79, 0.92, 1.32}
	charttmpirisAllreduce := chart.ContinuousSeries{
		Name:    "Allreduce",
		XValues: []float64{1, 2, 3, 4, 5, 6, 7, 8},
		YValues: irisAllreducetime,
	}
	Charts = append(Charts, charttmpirisAllreduce)

	irisSendRecvtime := []float64{1.59, 1.27, 1.12, 1.05, 1.08, 1.28, 2.06}
	charttmpirisSendRecv := chart.ContinuousSeries{
		Name:    "SendRecv",
		XValues: []float64{2, 3, 4, 5, 6, 7, 8},
		YValues: irisSendRecvtime,
	}
	Charts = append(Charts, charttmpirisSendRecv)

	graph := chart.Chart{
		Background: chart.Style{
			Padding: chart.Box{
				Top:  10,
				Left: 140,
			},
		},
		XAxis: chart.XAxis{
			Name: "Number of nodes",
		},
		YAxis: chart.YAxis{
			Name: "Training Time comsuming in minutes",
		},
		Series: Charts,
	}

	graph.Elements = []chart.Renderable{
		chart.LegendLeft(&graph),
	}
	graph.Height = 600
	f, _ := os.Create("irisSpendup.png")
	defer f.Close()
	graph.Render(chart.PNG, f)
}

func PlotIntelImgageSpeedup() {

	Charts := []chart.Series{}

	intelImAllreducetime := []float64{90, 74, 74, 69, 59, 60, 47, 44}
	charttmpAllreduce := chart.ContinuousSeries{
		Name:    "Allreduce",
		XValues: []float64{7, 8, 10, 12, 16, 20, 25, 30},
		YValues: intelImAllreducetime,
	}
	Charts = append(Charts, charttmpAllreduce)

	intelImSendRecvtime := []float64{203, 149, 118, 120, 111, 113, 138, 140}
	charttmpSendRecv := chart.ContinuousSeries{
		Name:    "SendRecv",
		XValues: []float64{7, 8, 10, 12, 16, 20, 25, 30},
		YValues: intelImSendRecvtime,
	}
	Charts = append(Charts, charttmpSendRecv)

	graph := chart.Chart{
		Background: chart.Style{
			Padding: chart.Box{
				Top:  10,
				Left: 140,
			},
		},
		XAxis: chart.XAxis{
			Name: "Number of nodes",
		},
		YAxis: chart.YAxis{

			Name: "Training Time comsuming in minutes",
		},
		Series: Charts,
	}

	graph.Elements = []chart.Renderable{
		chart.LegendLeft(&graph),
	}
	graph.Height = 600
	f, _ := os.Create("intelImageSpendup.png")
	defer f.Close()
	graph.Render(chart.PNG, f)
}
