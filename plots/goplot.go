package plots

import (
	"net/http"
	"os"
	"strconv"

	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/opts"
	"github.com/go-echarts/go-echarts/v2/types"
)

// generate random data for line chart
func generateLineItems() []opts.LineData {
	numEpochsenv, _ := strconv.Atoi(os.Getenv("numEpochs"))
	items := make([]opts.LineData, 0)

	// network.QAccuracy

	for i := 0; i < numEpochsenv; i++ {
		// items = append(items, opts.LineData{Value: Qdata[i]})
		items = append(items, opts.LineData{Value: i})
	}
	return items
}

func httpserver(w http.ResponseWriter, _ *http.Request) {
	// create a new line instance
	line := charts.NewLine()
	// set some global options like Title/Legend/ToolTip or anything else
	line.SetGlobalOptions(
		charts.WithInitializationOpts(opts.Initialization{Theme: types.ThemeWesteros}),
		charts.WithTitleOpts(opts.Title{
			Title:    "Line example in Westeros theme",
			Subtitle: "Line chart rendered by the http server this time",
		}))

	// Put data into instance
	numEpochsenv, _ := strconv.Atoi(os.Getenv("numEpochs"))

	XX := make([]string, numEpochsenv)
	for index := range XX {
		XX[index] = strconv.Itoa(index + 1)
	}

	// XAxis := []string{"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}

	line.SetXAxis(XX).
		AddSeries("Category A", generateLineItems()).
		//		AddSeries("Category B", generateLineItems()).
		SetSeriesOptions(charts.WithLineChartOpts(opts.LineChart{Smooth: true}))
	line.Render(w)
}

func PlotServer() {
	http.HandleFunc("/", httpserver)
	http.ListenAndServe(":8081", nil)
}
