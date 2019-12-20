function displayhistory(id,fData) {
    obj_home = document.getElementById("home_team").value;
    obj_away = document.getElementById("away_team").value;
    document.getElementById(id).innerHTML="";
    var count=0;
    document.getElementById(id).innerHTML+="<div id=\'histories\'>";
    var output="";
    for (i=0;i<fData[0].length;i++){
        if (count>=5){
            break;
        }
        if (obj_home===fData[0][i]["Home_team"]){
            if (obj_away===fData[0][i]["Away_team"]){
                count=count+1;
                output="<p>"+obj_home+" vs "+obj_away+fData[0][i]["Home_score"]+":"+fData[0][i]["Home_score"]+" in "+fData[0][i]["Year"]+" "+
                    fData[0][i]["Month"]+" "+fData[0][i]["Day"]+" "+"\n</p>";
                document.getElementById(id).innerHTML+=output;
            }
        }
        if (obj_away===fData[0][i]["Home_team"]){
            if (obj_home===fData[0][i]["Away_team"]){
                count=count+1;
                output="<p>"+obj_away+" vs "+obj_home+fData[0][i]["Home_score"]+":"+fData[0][i]["Home_score"]+" in "+fData[0][i]["Year"]+" "+
                    fData[0][i]["Month"]+" "+fData[0][i]["Day"]+" "+"\n</p>";
                document.getElementById(id).innerHTML+=output;
            }
        }
    }
    console.log(fData[0]);

    document.getElementById(id).innerHTML+="</div>";

}