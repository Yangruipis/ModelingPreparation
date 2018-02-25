// cd /home/ray/Documents/programme/stata
cd /home/ray/Documents/suibe/2017/建模/Modeling_Preparation/plot/datafile
use d_dijishi

spmap Mydata  using "c_dijishi" , id( ID_2 )clnumber(10) fcolor(Blues2)  ocolor(none ..) title("The Employment Status of Each City", size(*0.8)) subtitle("China 2015 " " ", size(*0.8)) legstyle(3) legend(ring(1) position(3))   plotregion(margin(vlarge))

use d_sheng1
spmap Mydata  using "c_sheng1" , id( ID_1 )clnumber(4) fcolor(Blues2)  ocolor(black ..) title("The Employment Status of Each City", size(*0.8)) subtitle("China 2015 " " ", size(*0.8)) legstyle(3) legend(ring(1) position(3))   plotregion(margin(vlarge))

//use d_seven
//spmap  var5 using "c_seven.dta", id(_ID) clnumber(7)  ocolor(none ..) fcolor(Blues) title("Pct. Catholics without reservations", size(*0.8)) subtitle("Italy, 1994-98" " ", size(*0.8))  diagram(variable(var5) range(0 2264) xcoord(var6) ycoord(var7) fcolor(red))


//use d_dijishi
//spmap GNLK using "c_dijishi" , id( _ID )clnumber(10) fcolor(Blues2)  ocolor(none ..)
//spmap GJLK using "c_dijishi" , id( _ID )clnumber(10) fcolor(Blues2)  ocolor(none ..)

//spmap gdp using "c_dijishi" , id( _ID )clnumber(4) fcolor(Blues2)  ocolor(black ..)
//spmap gdpper using "c_dijishi" , id( _ID )clnumber(4) fcolor(Blues2)  ocolor(black ..)
