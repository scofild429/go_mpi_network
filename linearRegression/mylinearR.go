package linearRegression

import (
	"fmt"
	"image/color"
	"log"
	"os"

	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func PrintSummary() {
	advertFile, err := os.Open("/home/si/go/src/github.com/scofild429/goai/datasets/Advertising.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer advertFile.Close()

	advertDF := dataframe.ReadCSV(advertFile)
	advertSummmary := advertDF.Describe()
	fmt.Println(advertSummmary)

}

func PrintDiagram() {
	advertFile, err := os.Open("/home/si/go/src/github.com/scofild429/goai/datasets/Advertising.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer advertFile.Close()

	advertDF := dataframe.ReadCSV(advertFile)

	for _, colName := range advertDF.Names() {
		plotVals := make(plotter.Values, advertDF.Nrow())
		for i, floatVal := range advertDF.Col(colName).Float() {
			plotVals[i] = floatVal
		}
		p := plot.New()

		p.Title.Text = fmt.Sprintf("Histogram of a %s", colName)
		h, err := plotter.NewHist(plotVals, 16)
		if err != nil {
			log.Fatal(err)
		}
		h.Normalize(1)
		p.Add(h)
		if err := p.Save(4*vg.Inch, 4*vg.Inch, colName+"_hist.png"); err != nil {
			log.Fatal(err)
		}
	}
}

func PrintCorelation() {
	advertFile, err := os.Open("/home/si/go/src/github.com/scofild429/goai/datasets/Advertising.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer advertFile.Close()

	advertDF := dataframe.ReadCSV(advertFile)
	yVals := advertDF.Col("Sales").Float()
	for _, colName := range advertDF.Names() {
		pts := make(plotter.XYs, advertDF.Nrow())
		for i, floatVal := range advertDF.Col(colName).Float() {
			pts[i].X = floatVal
			pts[i].Y = yVals[i]
		}
		p := plot.New()
		if err != nil {
			log.Fatal(err)

		}
		p.X.Label.Text = colName
		p.Y.Label.Text = "y"
		p.Add(plotter.NewGrid())

		s, err := plotter.NewScatter(pts)
		if err != nil {
			log.Fatal(err)
		}
		s.GlyphStyle.Color = color.RGBA{R: 255, B: 128, A: 255}
		s.GlyphStyle.Radius = vg.Points(3)

		p.Add(s)
		if err := p.Save(4*vg.Inch, 4*vg.Inch, colName+"_scatter.png"); err != nil {
			log.Fatal(err)
		}
	}
}
