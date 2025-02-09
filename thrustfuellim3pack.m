(* ::Package:: *)

(* ::Input::Initialization:: *)
(** This is code modified specifically for the 2d case **)
(** Set the directory that you want your plots to go to here **)
SetDirectory["/users/maxzweig/fuelthrust"];

Off[Solve::svars];
Off[FindRoot::lstol];
Off[InterpolatingFunction::dmval];
(** gets the axes for the energy limited reachable set associated with a given terminal time and thrust **)

getAxes[terminaltime_, quadform_, thrustmax_] := Module[{testbounds, teststats},
energysetaxes[quad_] :=
Eigenvectors[{Join[Join[IdentityMatrix[2], {{0,0,0,0},{0,0,0,0}}, 2], {{0,0,0, 0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0}}], quad[[7;; 12, 7 ;; 12]]}];
energyseteigenvalues[quad_] :=  Eigenvalues[{Join[Join[IdentityMatrix[2], {{0,0,0,0},{0,0,0,0}}, 2], {{0,0,0, 0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0}}], quad[[7;; 12, 7 ;; 12]]}];
vecs = energysetaxes[quadform];
vals = energyseteigenvalues[quadform];
ax1 = vecs[[1]][[1;;2]] * ((terminaltime thrustmax^2* vals[[1]]) // Sqrt);
ax2 = vecs[[2]][[1;;2]] * ((terminaltime thrustmax^2* vals[[2]]) // Sqrt);
{ax1,ax2}]

(** gets the minimum and maximum ratios between an energy limited set and an arbitrary reachable set given as set of points **)
reachablestatsoperator[points_,axis1_,axis2_] := Module[{rats, vecs}, 
ellipseVecFromUnitVec[uv_,ax1_, ax2_]:=1/Sqrt[(Normalize[ax1] . uv)^2/Norm[ax1]^2+(Normalize[ax2] . uv)^2/Norm[ax2]^2]*uv;
ratio[pos_]:= Norm[ellipseVecFromUnitVec[Normalize[pos[[1;;2]]],axis1, axis2] ]/ Norm[pos[[1;;2]]];
maxminrat[ v1s_] := {(ratio/@v1s)// Max, (ratio/@v1s)// Min};
toq2q1[vec_] := If[(vec[[1]] < 0 && vec[[2]] < 0) || (vec[[1]] >0 && vec[[2]] < 0), -vec, vec];
maxminvec[ v1s_] := {v1s[[ Position[(ratio/@v1s), (ratio/@v1s) // Max    ][[1,1]]]] // toq2q1 , v1s[[ Position[(ratio/@v1s), (ratio/@v1s) // Min   ][[1,1]]]] // toq2q1};
rats = maxminrat[ points];
vecs = maxminvec[ points];
{rats , {vecs[[1, 1;;2]], vecs[[2, 1;;2]]}}
] ;

(* computes minimum and maximum ratios between the thrust limited and thrust/fuel limited reachable sets, given as a set of points that are indexed by angle in their arrays.
Precondition is that no zeros or contained. *)
thrustfuelstats[thrustpoints_, fuelpoints_] := Module[{rats, vecs},

ListRatios[list1_, list2_] := MapThread[#1/#2&,{list1,list2}];
ratios = ListRatios[Map[Norm, thrustpoints], Map[Norm, fuelpoints]];
maxratio = ratios // Max;
minratio = ratios // Min;

toq2q1[vec_] := If[(vec[[1]] < 0 && vec[[2]] < 0) || (vec[[1]] >0 && vec[[2]] < 0), -vec, vec];

maxpos = Position[ratios, maxratio][[1,1]];
minpos = Position[ratios, minratio][[1,1]];

maxpos = fuelpoints[[maxpos]] // toq2q1;
minpos = fuelpoints[[minpos]] // toq2q1;

{maxratio, minratio, maxpos, minpos}
];
getFinalEnergyPos[tt_, tmax_, nt_, delta_] := Module[{},
energyform[tff_, na_] := Module[{c, r, sig, gamma, nmat},
c[n_] = {
{0, -2* n, 0, 1, 0, 0},
{2 * n, 0, 0, 0, 1, 0},
{0, 0, 0, 0,0 , 1},
{-1, 0, 0, 0, 0, 0},
{0, -1, 0, 0,0,0 }, 
{0, 0, -1, 0, 0, 0 }
}; (*Good*)
c[n] // MatrixForm;
r[t_, h_] = {{72 t Cos[h t]+(154 h t+48 h^3 t^3-384 Sin[h t]-27 Sin[2 h t])/(4 h),-3 h t^2+(6-6 Cos[h t])/h,0,(8+12 h^2 t^2-8 Cos[h t]-24 h t Sin[h t]+9 Sin[h t]^2)/(2 h^2),(23 h t+6 h^3 t^3+42 h t Cos[h t]-56 Sin[h t]-9/2 Sin[2 h t])/h^2,0},{-3 h t^2+(6-6 Cos[h t])/h,t,0,(2 (-h t+Sin[h t]))/h^2,-((3 t^2)/2)+(4-4 Cos[h t])/h^2,0},{0,0,(2 h t+Sin[2 h t])/(4 h),0,0,Sin[h t]^2/(2 h^2)},{(8+12 h^2 t^2-8 Cos[h t]-24 h t Sin[h t]+9 Sin[h t]^2)/(2 h^2),(2 (-h t+Sin[h t]))/h^2,0,(26 h t-32 Sin[h t]+3 Sin[2 h t])/(4 h^3),(3 (-h t+Sin[h t])^2)/h^3,0},{(23 h t+6 h^3 t^3+42 h t Cos[h t]-56 Sin[h t]-9/2 Sin[2 h t])/h^2,-((3 t^2)/2)+(4-4 Cos[h t])/h^2,0,(3 (-h t+Sin[h t])^2)/h^3,(14 h t+3 h^3 t^3+24 h t Cos[h t]-32 Sin[h t]-3 Sin[2 h t])/h^3,0},{0,0,Sin[h t]^2/(2 h^2),0,0,-((-2 h t+Sin[2 h t])/(4 h^3))}};(*Good*)
r[t, n] // MatrixForm;
sig[t_, n_] = {{4-3 Cos[n t],0,0,Sin[n t]/n,(2-2 Cos[n t])/n,0},{6 (-n t+Sin[n t]),1,0,(2 (-1+Cos[n t]))/n,-3 t+(4 Sin[n t])/n,0},{0,0,Cos[n t],0,0,Sin[n t]/n},{3 n Sin[n t],0,0,Cos[n t],2 Sin[n t],0},{6 n (-1+Cos[n t]),0,0,-2 Sin[n t],-3+4 Cos[n t],0},{0,0,-n Sin[n t],0,0,Cos[n t]}}; (* Good *)
gamma[t_, t0_, n_] = sig[t, n] . (sig[t0, n] // Inverse);
nmat[t0_, tf_, n_] = (Join[sig[t0, n] // Inverse, - sig[tf, n] // Inverse, 2] // Transpose) . (c[n] // Transpose) . (r[tf, n] // Inverse) . c[n] . Join[sig[t0, n] // Inverse, - sig[tf, n] // Inverse, 2];
nmat[0, tff, na]
];

udd = IdentityMatrix[6] - Outer[Times, Join[{delta[[2]], -delta[[1]]}, {0,0,0,0}], Join[{delta[[2]], -delta[[1]]}, {0,0,0,0}]];
basis = Orthogonalize[udd];
projmat = Join[{basis[[1]]}, basis[[3;;]]] // Transpose;
projmatT = projmat // Transpose;
selectionMat = {{1,0,0,0,0,0}, {0,1,0,0,0,0}, {0,0,0,0,0,0}, {0,0,0,0,0,0}, {0,0,0,0,0,0}, {0,0,0,0,0,0}};
eformxf = energyform[tt, nt][[7;;12, 7;;12]];
ydir = Eigenvectors[{projmatT . selectionMat . projmat, projmatT . eformxf . projmat}][[1]];

xdir = projmat . ydir;
tmag = (tt tmax^2 / (xdir . eformxf . xdir)) // Sqrt;
xdir = If[xdir[[1]] delta[[1]] < 0, -xdir, xdir];

Print["energy computation executed"];
xdir tmag
];
propStateCostatev2[ics_, isp_, tf_] := Module[{vars, dyn},
mag[lxv_, lyv_, lzv_] :=(lxv ^ 2+ lyv ^2 + lzv ^2) // Sqrt;
switchingfunc [lxv_, lyv_, lzv_, lmag_, mas_, c_] := mag[lxv, lyv, lzv] - lmag mas /c; 
vars={x,y,z,vx,vy,vz,m,lxr,lyr,lzr,lxv,lyv,lzv,lm};
eqnsOpt = Join[(- A // Transpose) . {lxr, lyr, lzr, lxv, lyv, lzv}, {( 1/2 +Sign[switchingfunc[lxv, lyv, lzv, lm, m, isp]] / 2)tmax mag[lxv, lyv, lzv]  / (m ^2)}];
eqnsStateOpt={  2 vy n + 3n^2 x,    -2 n vx, - n^2 z  } + (1/m)( 1/2 +Sign[switchingfunc[lxv, lyv, lzv, lm, m, isp]] / 2) tmax {lxv, lyv, lzv} / mag[lxv, lyv, lzv];
dyn=Join[Join[{vx,vy,vz},Join[eqnsStateOpt, {- ( 1/2 +Sign[switchingfunc[lxv, lyv, lzv , lm, m, isp]] / 2)tmax/isp}]],eqnsOpt];

soln =  NDSolveValue[
Join[Thread[(#'[t]&/@vars[[1;;14]])==(dyn/.Thread[vars->(#[t]&/@vars)])],
Thread[(#[0]&/@vars)==ics]],
vars,  {t, tf, tf}
];
#[tf]&/@soln
]
findInitialCostates[icRel_, fcRel_, n_, A_, td_, mmax_, m0_, tmax_] := Module[{vars, dyn},
eqnsStateOpt={  2 vy n + 3n^2 x,    -2 n vx, - n^2 z  } - {lxv, lyv, lzv};
eqnsOpt = (- A // Transpose) . {lxr, lyr, lzr, lxv, lyv, lzv};
getInitCostates[varMat_,icsRel_,fcsRel_]:=LinearSolve[varMat[[1;;6,7;;12]],fcsRel-varMat[[1;;6,1;;6]] . icsRel];
topLeftMatrix=A;
topRightMatrix=ConstantArray[0,{3,6}];
middleLeftMatrix=ConstantArray[0,{3,3}];
middleRightMatrix=-IdentityMatrix[3];
     bottomRightMatrix=-A // Transpose;
bottomLeftMatrix=ConstantArray[0,{6,6}];
fullmatrix = Join[Join[A, Join[ topRightMatrix, Join[middleLeftMatrix, middleRightMatrix, 2]], 2], Join[bottomLeftMatrix, bottomRightMatrix, 2 ] ];
stm[tf_] := MatrixExp[fullmatrix tf];
initCo = getInitCostates[stm[td], icRel, fcRel];
lambdastm[ta_] = MatrixExp[(-A // Transpose) ta];
lma = isp ((lambdastm[mmax isp / tmax] . initCo)[[4;;6]]   // Norm)/(m0 - mmax) - NIntegrate[  tmax (((lambdastm[t] . initCo)[[4;;6]]) // Norm) / ((m0 - tmax t / isp) ^2), {t, 0, mmax isp / tmax}];
initCoM = Join[initCo, {lma}];
initCoM
]
rootfind[init_, mm_, direction_, A_, m0_, isp_, timeLength_] := Module[{},
s = MatrixExp[-(A //Transpose) timeLength];
ly = init[[2]];
subs = Solve[s[[4;;5]] . {v1, v2, v3, v4, v5, v6} =={0,0}, {v1, v2, v3, v4, v5, v6}][[1]];
falt[joint_]:= Module[{out},
out= propStateCostatev2[Join[{0,0,0,0,0,0,m0},   Join[Join[{joint[[1]] , ly, 0}, ({v4, v5} /. subs ) /. {v1->joint[[1]], v2 ->ly} ], {0, joint[[2]]}]], isp, timeLength];
cons[dey_, lf_, xf_, mmax_] := Module[{},
udd = (2 // IdentityMatrix) - ({dey} // Transpose) . {dey};
c1 ={ (udd . {xf[[1]], xf[[2]]})[[1]] } ;
c2 =  If [timeLength tmax / isp<=mmax, {(m0 - xf[[7]]) -timeLength tmax / isp}, {(m0 - xf[[7]]) -mmax}];
Join[c1, c2] // Flatten
];
cons[direction, out[[8;;14]], out[[1;;7]] , mm]
];
Print["will the kernel crash?"];
poss = FindRoot[falt[l], {{l, {init[[1]], init[[3]]} }}, Evaluated->False, PrecisionGoal->20];
Print["it didn't"];

(** if poss does not satisfy the directional and mass constraints within satisfaction, try again with the two other possible values for lambda y, and pick the best **)
{poss[[1, 2, 1]], ly, poss[[1, 2, 2]]}
]
rootfindalt2[init_, mm_, direction_, A_, m0_, isp_, timeLength_] := Module[{},
s = MatrixExp[-(A //Transpose) timeLength];
lx = init[[1]];
subs = Solve[s[[4;;5]] . {v1, v2, v3, v4, v5, v6} =={0,0}, {v1, v2, v3, v4, v5, v6}][[1]];
falt[joint_]:= Module[{out},
out= propStateCostatev2[Join[{0,0,0,0,0,0,m0},   Join[Join[{lx , joint[[1]], 0}, ({v4, v5} /. subs ) /. {v1->lx, v2 ->joint[[1]]} ], {0, joint[[2]]}]], isp, timeLength];
cons[dey_, lf_, xf_, mmax_] := Module[{},
udd = (2 // IdentityMatrix) - ({dey} // Transpose) . {dey};
c1 ={ (udd . {xf[[1]], xf[[2]]})[[1]] } ;
c2 =  If [timeLength tmax / isp<=mmax, {(m0 - xf[[7]]) -timeLength tmax / isp}, {(m0 - xf[[7]]) -mmax}];
Join[c1, c2] // Flatten
];
cons[direction, out[[8;;14]], out[[1;;7]] , mm]
];
poss = FindRoot[falt[l], {{l, {init[[2]], init[[3]]} }}, Evaluated->False, PrecisionGoal->20];

(** if poss does not satisfy the directional and mass constraints within satisfaction, try again with the two other possible values for lambda y, and pick the best **)
{lx, poss[[1, 2, 1]], poss[[1, 2, 2]]}
]

getReachablePoint[direction_, endTime_, tmax_, n_, m0_, A_, fuelpercent_, isp_] := Module[{},
posf = getFinalEnergyPos[endTime, tmax, n, direction];
s = MatrixExp[-(A //Transpose) endTime];
subs = Solve[s[[4;;5]] . {v1, v2, v3, v4, v5, v6} =={0,0}, {v1, v2, v3, v4, v5, v6}][[1]];
initCostates = findInitialCostates[{0,0,0,0,0,0}, posf, n, A, endTime, tmax fuelpercent endTime / isp, m0, tmax / m0 ];
initialGuess2 = Join[{initCostates[[1]], initCostates[[2]]},{initCostates[[7]]}];
lxm2 = rootfind[initialGuess2,  tmax fuelpercent endTime / isp, direction // N, A, m0, isp, endTime];
out= propStateCostatev2[Join[{0,0,0,0,0,0,m0},   Join[Join[{lxm2[[1]], lxm2[[2]], 0}, ({v4, v5} /. subs ) /. {v1->lxm2[[1]], v2 ->lxm2[[2]]}, {0, lxm2[[3]]}]]], isp, endTime];
Print[out[[1;;2]]];
out[[1;;2]]
]

solveBVPsNested[initialAngle_, initialGuess_, endTime_, tmax_, n_, m0_, A_, fuelpercent_, isp_, numPoints_] := Module[{},
(** returns the direction the initial costate was computed in, as well as the initial costate 
	initialGuess contains all of the costates. 
  If prevDirectionCostate costate is identically 0, do the energy computation to obtain an initial guess. 
  Homotopy from delta to delta to travese the ring. If / when constraint isn't satisfied, perform the energy computation to obtain a new initial guess and try again.
**)
bvpNester[prevAngleCostatePosition_] := Module[{},
prevAngle = prevAngleCostatePosition[[1]];
prevCostate = prevAngleCostatePosition[[2]];
prevPosition = prevAngleCostatePosition[[3]];

nextAngle = prevAngle + 2 * Pi / numPoints;
nextDirection = {Cos[nextAngle], Sin[nextAngle]} // N;

nextCostate = If[prevCostate == {0,0,0}, findInitialCostates[{0,0,0,0,0,0}, getFinalEnergyPos[endTime, tmax, n, nextDirection], n, A, endTime, tmax fuelpercent endTime / isp, m0, tmax / m0 ][[{1, 2, 7}]], prevCostate];

nextGuess = {nextCostate[[1]],nextCostate[[2]], nextCostate[[3]]};
lxm2 = rootfind[nextGuess,  tmax fuelpercent endTime / isp, nextDirection, A, m0, isp, endTime];
out= propStateCostatev2[Join[{0,0,0,0,0,0,m0},   Join[Join[{lxm2[[1]], lxm2[[2]], 0}, ({v4, v5} /. subs ) /. {v1->lxm2[[1]], v2 ->lxm2[[2]]}, {0, lxm2[[3]]}]]], isp, endTime];

terminalAngle = If[out[[1]] == 0 && out[[2]] == 0, 10000, ArcTan[out[[1]], out[[2]]]];
terminalMass = m0 - out[[7]];

(** retrying with energy optimal initial guess if necessary**)
 nextGuessReturn = If[Abs[(terminalAngle + Pi) - Mod[nextAngle, 2 Pi]] < 0.0025 && Abs[terminalMass / (tmax fuelpercent endTime / isp) - 1] < 0.01,{lxm2[[1]], lxm2[[2]], lxm2[[3]]}, 
rootfind[findInitialCostates[{0,0,0,0,0,0}, getFinalEnergyPos[endTime, tmax, n, nextDirection], n, A, endTime, tmax fuelpercent endTime / isp, m0, tmax / m0 ][[{1, 2, 7}]],  tmax fuelpercent endTime / isp, nextDirection, A, m0, isp, endTime ]
];

out2 = propStateCostatev2[Join[{0,0,0,0,0,0,m0},   Join[Join[{nextGuessReturn[[1]], nextGuessReturn[[2]], 0}, ({v4, v5} /. subs ) /. {v1->nextGuessReturn[[1]], v2 ->nextGuessReturn[[2]]}, {0, nextGuessReturn[[3]]}]]], isp, endTime];

(** if still failure to perfectly satisfy constraints, retrying with alternate optimization holding x constant and allowing changes in y **)
terminalAngle = If[out2[[1]] == 0 && out2[[2]] == 0, 10000, ArcTan[out2[[1]], out2[[2]]]];
If[ (out2[[1]] == 0  && out2[[2]] ==0)  || Abs[terminalMass / (tmax fuelpercent endTime / isp) - 1] >= 0.001 || Abs[(terminalAngle + Pi) - Mod[nextAngle, 2 Pi]] >= 0.0025,
( res = rootfindalt2[{nextGuessReturn[[1]], nextGuessReturn[[2]], nextGuessReturn[[3]]},  tmax fuelpercent endTime / isp, nextDirection, A, m0, isp, endTime];
out3 = propStateCostatev2[Join[{0,0,0,0,0,0,m0},   Join[Join[{res[[1]], res[[2]], 0}, ({v4, v5} /. subs ) /. {v1->res[[1]], v2 ->res[[2]]}, {0, res[[3]]}]]], isp, endTime];
{nextAngle, res, out3[[1;;2]], out3[[7]]}
),{nextAngle, nextGuessReturn, out2[[1;;2]], out2[[7]]}
]
];
NestList[bvpNester, {initialAngle, initialGuess, {0,0,0,0,0,0,0}}, numPoints]
]


(* ::Input::Initialization:: *)
mu1 = 3.986 10^14 ;
endTime = 3600;
alpha = 7780000;
n = (mu1 / alpha^3)^(1/2);
A = {{0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1,0}, {0,0,0,0,0,1}, {3 n^2, 0, 0, 0, 2n, 0}, {0, 0, 0, -2n, 0,0}, {0, 0, -n^2, 0, 0, 0}};
isp = 10000;
tmax = 0.002;
mmax = 0.002 endTime / isp;
m0 = 1;


(* ::Input::Initialization:: *)
computestats[thrustmax_, n_, endTime_, m0_, fuelpercent_, isp_, num_] := Module[{testbounds, teststats},
Print[fuelpercent];
A = {{0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1,0}, {0,0,0,0,0,1}, {3 n^2, 0, 0, 0, 2n, 0}, {0, 0, 0, -2n, 0,0}, {0, 0, -n^2, 0, 0, 0}};
testbounds = solveBVPsNested[Pi/4, {0,0,0}, endTime, thrustmax, n, m0, A, fuelpercent, isp, num][[2;;]];
testbounds2 = solveBVPsNested[Pi/4, {0,0,0}, endTime, thrustmax, n, m0, A, 1, isp, num][[2;;]];

testmasses = testbounds[[;;,4]];
testmasses2= testbounds2[[;;, 4]];
testxfs = testbounds[[;;, 3]];
testxfs2 = testbounds2[[;;, 3]];

removalIndices = Position[Map[Norm, testxfs],_?((#!=0)&)] // Flatten;
augRemIndices = Position[Map[Norm, testxfs2],_?((#!=0)&)] // Flatten;
massRemIndices = Position[testmasses, _? (( Abs[#/ ((thrustmax fuelpercent endTime / isp) - m0)] - 1 <= 0.001) &)] // Flatten;
augMessRemIndices = Position[testmasses2, _? (( Abs[#/ ((thrustmax endTime / isp) - m0)] - 1<= 0.001) &)] // Flatten;

rats = MapThread[Abs[ArcTan[#1[[1]], #1[[2]]] - ArcTan[#2[[1]],#2[[2]]]] &, {testxfs, testxfs2}];
addRem = Position[rats, _? (( Abs[#] < 0.01) &)] //Flatten;

remInds = Intersection[augRemIndices, removalIndices, massRemIndices, augMessRemIndices, addRem];

thrustfuelstats[testxfs[[remInds]], testxfs2[[remInds]]]
]


(** plots the thrust or fuel limited reachable set alone.**)
plotthrustfuel[thrustmax_, n_, endTime_, m0_, fuelpercent_, isp_, num_] := Module[{testbounds, teststats},

A = {{0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1,0}, {0,0,0,0,0,1}, {3 n^2, 0, 0, 0, 2n, 0}, {0, 0, 0, -2n, 0,0}, {0, 0, -n^2, 0, 0, 0}};
testbounds = solveBVPsNested[Pi/4, {0,0,0}, endTime, thrustmax, n, m0, A, fuelpercent, isp, num][[2;;]];
testbounds2 = solveBVPsNested[Pi/4, {0,0,0}, endTime, thrustmax, n, m0, A, 1, isp, num][[2;;]];

testmasses = testbounds[[;;,4]];
testmasses2= testbounds2[[;;, 4]];
testxfs = testbounds[[;;, 3]];
testxfs2 = testbounds2[[;;, 3]];

removalIndices = Position[Map[Norm, testxfs],_?((#!=0)&)] // Flatten;
augRemIndices = Position[Map[Norm, testxfs2],_?((#!=0)&)] // Flatten;
massRemIndices = Position[testmasses, _? (( Abs[#/ ((thrustmax fuelpercent endTime / isp) - m0)] - 1 <= 0.001) &)] // Flatten;
augMessRemIndices = Position[testmasses2, _? (( Abs[#/ ((thrustmax endTime / isp) - m0)] - 1<= 0.001) &)] // Flatten;

rats = MapThread[Abs[ArcTan[#1[[1]], #1[[2]]] - ArcTan[#2[[1]],#2[[2]]]] &, {testxfs, testxfs2}];
addRem = Position[rats, _? (( Abs[#] < 0.01) &)] //Flatten;

remInds = Intersection[augRemIndices, removalIndices, massRemIndices, augMessRemIndices, addRem];

{maxratio, minratio, maxpos, minpos} = thrustfuelstats[testxfs[[remInds]], testxfs2[[remInds]]];
Print[maxratio, " ", minratio, " ", maxpos[[1]] // Normalize, " ", maxpos[[2]] // Normalize, " ", minpos[[1]] // Normalize, " ", minpos[[2]] // Normalize];

testbounds = Join[testxfs, testxfs2]; 

flipTransformation=ReflectionTransform[{1,-1}];
plot =  Legended[Show[Graphics[GeometricTransformation[GeometricTransformation[Point[testbounds[[;;, 1 ;; 2]]],flipTransformation],ScalingTransform[{0.001, 0.001}]]], Graphics[{Red, GeometricTransformation[Arrow[{{0,0},0.001maxpos[[1]]} ], flipTransformation]}], Graphics[{Red, GeometricTransformation[Arrow[{{0,0},-0.001 maxpos[[1]]} ], flipTransformation]}], Graphics[{Blue, GeometricTransformation[Arrow[{{0,0},0.001minpos[[1]]} ], flipTransformation]}],
Graphics[{Blue, GeometricTransformation[Arrow[{{0,0},-0.001 minpos[[1]]} ], flipTransformation]}], 
Axes -> True, ImagePadding ->30,   PlotLabel -> "Percent Fuel Limited RRS", AxesLabel -> {"y", "x"}], Placed[LineLegend[{Blue,Red},{"Minimum Ratio","Maximum Ratio"}],{0.1,0.001}]] ;
Export[StringForm["RRSttt``.png", endTime] // ToString, plot]
];



(* ::Input::Initialization:: *)
mu1 = 3.986 10^14 ;
endTime = 360;
alpha = 7780000;
n = (mu1 / alpha^3)^(1/2);
isp = 10000;
tmax = 0.002;
m0 = 1;
fuelpercent =0.675;


(* ::Input::Initialization:: *)
(**plotthrustfuel[tmax,n, endTime, m0, fuelpercent, isp, 700]**)
computestats[tmax, n, endTime, m0, fuelpercent, isp, 100]


(* ::Input::Initialization:: *)
numfuelsamples=5;
stats = Table[computestats[tmax, n, 3600, m0, fuelval, isp, 60], {fuelval, 0.675, 1, (1 - 0.675) / numfuelsamples}];
maxratios = stats[[;;, 1]];
minratios = stats[[;;, 2]];



(* ::Input::Initialization:: *)
Print[maxratios];
Print[minratios];
Range[0.675, 1, (1 - 0.675) / numfuelsamples] 
l1 = ListLinePlot[{{Range[0.675, 1, (1 - 0.675) / numfuelsamples], maxratios} // Transpose,{Range[0.675, 1, (1 - 0.675) / numfuelsamples], minratios} // Transpose}, AxesLabel-> {"Fuel Usage", "Ratio" } , PlotLegends->LineLegend[{"Maximum Ratio","Minimum Ratio"}]]  
Export["fuelmaxrats.png", l1]



(* ::Input::Initialization:: *)
stats = Table[computestats[tmax, n, tval, m0, fuelpercent, isp, 60], {tval, 60, 8060, 1000}];
maxratios = stats[[;;, 1]];
minratios = stats[[;;, 2]];
Print[maxratios];
Print[minratios];
Print[Range[60, 8600,1000]];


(* ::Input::Initialization:: *)
l2 = ListLinePlot[{{Range[60, 8060,1000], maxratios} // Transpose,{Range[60, 8060,1000], minratios} // Transpose}, AxesLabel-> {"Terminal Time", "Ratio" } , PlotLegends->LineLegend[{"Maximum Ratio","Minimum Ratio"}]]  
Export["fuelmaxrats.png", l2]
