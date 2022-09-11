module github.com/scofild429/goai/goai

go 1.18

replace github.com/scofild429/goai/myData => ../myData

require (
	github.com/joho/godotenv v1.4.0
	github.com/sbromberger/gompi v0.2.0
	github.com/scofild429/goai/myData v0.0.0-00010101000000-000000000000
	github.com/scofild429/goai/network v0.0.0-00010101000000-000000000000
	github.com/wcharczuk/go-chart/v2 v2.1.0
)

require (
	github.com/deckarep/golang-set v1.8.0 // indirect
	github.com/go-gota/gota v0.12.0 // indirect
	github.com/golang/freetype v0.0.0-20170609003504-e2365dfdc4a0 // indirect
	github.com/gonum/floats v0.0.0-20181209220543-c233463c7e82 // indirect
	github.com/gonum/internal v0.0.0-20181124074243-f884aa714029 // indirect
	golang.org/x/image v0.0.0-20220722155232-062f8c9fd539 // indirect
	golang.org/x/net v0.0.0-20210423184538-5f58ad60dda6 // indirect
	gonum.org/v1/gonum v0.11.0 // indirect
)

replace github.com/scofild429/goai/linearRegression => ../linearRegression

replace github.com/scofild429/goai/logisticRegression => ../logisticRegression

replace github.com/scofild429/goai/network => ../network

replace github.com/scofild429/goai/plots => ../plots
