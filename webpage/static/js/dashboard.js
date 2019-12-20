//function在js里面代表代码块。没有参数的话直接执行。
//js中的类型有：值类型：字符串，数字，布尔， 引用型：对象，数组，函数
//用new创建一个对象

var count=0;

function dashboard(id, fData) {
    // Define the color to change if your mouse move on the bar
    // ID 是之前在html里的div（分块）
    //{"data":[{home:..., away:..., prediction:..., percent{win lose draw}}]}
    //这里测验一个对象。要访问这个对象的属性，可以person.id 或者person["id"]
    //对象可以有方法

    var person={
        firstname:"Jin",
        lastname:"Huang",
        id:"jh4137",
        fullname: function () {
            return this.firstname+" "+this.lastname+ " "+ this.id;
        }
    };


    console.log(fData.length);

   //去重函数unique
    function unique(arr){
        var hash=[];
        for (var i = 0; i < arr.length; i++) {
            if(hash.indexOf(arr[i])===-1){
                hash.push(arr[i]);
            }
        }
        return hash;
    }

    //处理home_teams 和away_teams
    var home_team=[];
    var away_team=[];
    for (i=0;i<fData.length;i++){
        home_team.push(fData[i]["Home"]);
        away_team.push(fData[i]["Away"]);
    }

    home_team=unique(home_team);
    away_team=unique(away_team);


    document.write("<div id=\"buttons\">");

    document.write("<label for='home_team'> Home Team </label>")

    document.write(`<select name='home team' id= "home_team">`);
    for (i=0;i<home_team.length;i++){
        document.write(
        "<option value="+ home_team[i]+">"+ home_team[i] +"</option>");
    }




    document.write("</select>\n");

    //处理away_teams
    document.write("<label for='away_team'> Away Team </label>")
    document.write("<select name='away team' id= \"away_team\""  + ">");
    for (i=0;i<away_team.length;i++){
        document.write(
        "<option value="+ away_team[i]+">"+ away_team[i] +"</option>");
    }
    document.write("</select>\n");
    document.write("<input type=\'button\' value=\'query\' onClick=\"fun(\'#dashboard\',Data);displayhistory(\'displayhistory\',History)\"/>")


}
    document.write("</div>");

    function fun(id, fData){

        obj_home = document.getElementById("home_team");
        obj_away = document.getElementById("away_team");
        //alert(obj_home.value+","+obj_away.value);

        //找到相对应的数据点
        var index=-1;


        for(i=0;i<fData.length;i++){
            if (fData[i]["Home"]===obj_home.value){
                if (fData[i]["Away"]===obj_away.value){
                    index=i;
                    break;
                }
            }
        }


        console.log(obj_away.value);
        if (index===-1){
            alert("no_result");
        }
        else{alert(fData[index]["Prediction"])}

        var tF = ['Win', 'Lose',　'Draw'].map(function (d) {
            let Percent;
            return {
                type: d, count: fData[index]['Percent'][d]
            }
        });
        //tF:{Win:33,"Lose":34,"Draw":33}


        function segColor(c) {
            cmap = {
                Win: "orange",
                Draw: "violet",
                Lose: "blue"
            };
            /* TO FINISH */
            return cmap[c];
        }



        function pieChart(pD) {

            var pC = {}, pieDim = {w: 280, h: 530};
            pieDim.r = Math.min(pieDim.w, pieDim.h) / 2;

            // create svg for pie chart.
            var piesvg = d3.select(id).append("svg")
                .attr("width", pieDim.w).attr("height", pieDim.h).append("g")
                .attr("transform", "translate(" + pieDim.w /2 + "," + pieDim.h / 2 + ")");

            // create function to draw the arcs of the pie slices.
            var arc = d3.svg.arc().outerRadius(pieDim.r - 10).innerRadius(0);

            // create a function to compute the pie slice angles.
            var pie = d3.layout.pie().sort(null).value(function (d) {
                return d.count;
            });


            // Draw the pie slices.
            piesvg.selectAll("path").data(pie(pD)).enter().append("path").attr("d", arc)
                .each(function (d) {
                    this._current = d;
                })
                .style("fill",
                    function (d) {
                    return segColor(d.data.type);
                });


                return pC;
        }







        function legend(lD) {
        var leg = {};

        // create table for legend.
        var legend = d3.select(id).append("table").attr('class', 'legend');

        // create one row per segment.
        var tr = legend.append("tbody").selectAll("tr").data(lD).enter().append("tr");

        // create the first column for each segment.
        tr.append("td").append("svg").attr("width", '16').attr("height", '16').append("rect")
            .attr("width", '16').attr("height", '16')
            .attr("fill", function (d) {
                return segColor(d.type);
            });

        // create the second column for each segment.
        tr.append("td").text(function (d) {
            return d.type;
        });

        // create the third column for each segment.
        tr.append("td").attr("class", 'legendFreq')
            .text(function (d) {
                return d3.format(",")(d.count);
            });

        // create the fourth column for each segment.
        tr.append("td").attr("class", 'legendPerc')
            .text(function (d) {
                return getLegend(d, lD);
            });


        function getLegend(d, aD) { // Utility function to compute percentage.
            return d3.format("%")(d.count / d3.sum(aD.map(function (v) {
                return v.count;
            })));
        }

        return leg;
    }

        d3.select(id).selectAll("svg").remove();
        d3.select(id).selectAll("td").remove();
        pieChart(tF);
        legend(tF)






    }