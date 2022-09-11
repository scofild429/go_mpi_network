package logisticRegression

import (
	"image/color"
	"log"
	"math"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func LGplotmuster() {
	p := plot.New()
	p.Title.Text = "Logistic Regression"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "f(x)"

	logisticPlotter := plotter.NewFunction(func(x float64) float64 { return logistic(x) })
	logisticPlotter.Color = color.RGBA{B: 255, A: 255}

	p.Add(logisticPlotter)
	p.X.Min = -10
	p.X.Max = 10
	p.Y.Min = -0.1
	p.Y.Max = 1.1

	if err := p.Save(4*vg.Inch, 4*vg.Inch, "logistic.png"); err != nil {
		log.Fatal(err)
	}
}
func logistic(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
