
I
X_inputPlaceholder*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
E
PlaceholderPlaceholder*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

8
Placeholder_1Placeholder*
dtype0*
shape:
S
truncated_normal/shapeConst*%
valueB"            *
dtype0
B
truncated_normal/meanConst*
valueB
 *    *
dtype0
D
truncated_normal/stddevConst*
valueB
 *ÍÌÌ=*
dtype0
z
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*

seed *
seed2 *
dtype0*
T0
_
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0
M
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0
d
Variable
VariableV2*
shape:*
dtype0*
	container *
shared_name 

Variable/AssignAssignVariabletruncated_normal*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable
I
Variable/readIdentityVariable*
T0*
_class
loc:@Variable
6
zerosConst*
valueB*    *
dtype0
Z

Variable_1
VariableV2*
shape:*
dtype0*
	container *
shared_name 

Variable_1/AssignAssign
Variable_1zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_1
O
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1
U
truncated_normal_1/shapeConst*%
valueB"            *
dtype0
D
truncated_normal_1/meanConst*
valueB
 *    *
dtype0
F
truncated_normal_1/stddevConst*
valueB
 *ÍÌÌ=*
dtype0
~
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*

seed *
seed2 *
dtype0*
T0
e
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0
S
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0
f

Variable_2
VariableV2*
shape:*
dtype0*
	container *
shared_name 

Variable_2/AssignAssign
Variable_2truncated_normal_1*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_2
O
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2
8
zeros_1Const*
valueB*    *
dtype0
Z

Variable_3
VariableV2*
shape:*
dtype0*
	container *
shared_name 

Variable_3/AssignAssign
Variable_3zeros_1*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_3
O
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3
U
truncated_normal_2/shapeConst*%
valueB"            *
dtype0
D
truncated_normal_2/meanConst*
valueB
 *    *
dtype0
F
truncated_normal_2/stddevConst*
valueB
 *ÍÌÌ=*
dtype0
~
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*

seed *
seed2 *
dtype0*
T0
e
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0
S
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0
f

Variable_4
VariableV2*
shape:*
dtype0*
	container *
shared_name 

Variable_4/AssignAssign
Variable_4truncated_normal_2*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_4
O
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4
8
zeros_2Const*
valueB*    *
dtype0
Z

Variable_5
VariableV2*
shape:*
dtype0*
	container *
shared_name 

Variable_5/AssignAssign
Variable_5zeros_2*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_5
O
Variable_5/readIdentity
Variable_5*
T0*
_class
loc:@Variable_5
M
truncated_normal_3/shapeConst*
valueB"     *
dtype0
D
truncated_normal_3/meanConst*
valueB
 *    *
dtype0
F
truncated_normal_3/stddevConst*
valueB
 *ÍÌÌ=*
dtype0
~
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*

seed *
seed2 *
dtype0*
T0
e
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0
S
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0
`

Variable_6
VariableV2*
shape:
*
dtype0*
	container *
shared_name 

Variable_6/AssignAssign
Variable_6truncated_normal_3*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_6
O
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6
9
zeros_3Const*
valueB*    *
dtype0
[

Variable_7
VariableV2*
shape:*
dtype0*
	container *
shared_name 

Variable_7/AssignAssign
Variable_7zeros_3*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_7
O
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7
M
truncated_normal_4/shapeConst*
valueB"   
   *
dtype0
D
truncated_normal_4/meanConst*
valueB
 *    *
dtype0
F
truncated_normal_4/stddevConst*
valueB
 *ÍÌÌ=*
dtype0
~
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*

seed *
seed2 *
dtype0*
T0
e
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
T0
S
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
T0
_

Variable_8
VariableV2*
shape:	
*
dtype0*
	container *
shared_name 

Variable_8/AssignAssign
Variable_8truncated_normal_4*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_8
O
Variable_8/readIdentity
Variable_8*
T0*
_class
loc:@Variable_8
8
zeros_4Const*
valueB
*    *
dtype0
Z

Variable_9
VariableV2*
shape:
*
dtype0*
	container *
shared_name 

Variable_9/AssignAssign
Variable_9zeros_4*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_9
O
Variable_9/readIdentity
Variable_9*
T0*
_class
loc:@Variable_9

Conv2DConv2DX_inputVariable/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC
,
addAddConv2DVariable_1/read*
T0

ReluReluadd*
T0

Conv2D_1Conv2DReluVariable_2/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC
0
add_1AddConv2D_1Variable_3/read*
T0

Relu_1Reluadd_1*
T0
u
MaxPoolMaxPoolRelu_1*
T0*
ksize
*
strides
*
paddingSAME*
data_formatNHWC
8
dropout/ShapeShapeMaxPool*
T0*
out_type0
G
dropout/random_uniform/minConst*
valueB
 *    *
dtype0
G
dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0
s
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape*

seed *
seed2 *
dtype0*
T0
b
dropout/random_uniform/subSubdropout/random_uniform/maxdropout/random_uniform/min*
T0
l
dropout/random_uniform/mulMul$dropout/random_uniform/RandomUniformdropout/random_uniform/sub*
T0
^
dropout/random_uniformAdddropout/random_uniform/muldropout/random_uniform/min*
T0
B
dropout/addAddPlaceholder_1dropout/random_uniform*
T0
,
dropout/FloorFloordropout/add*
T0
7
dropout/divRealDivMaxPoolPlaceholder_1*
T0
7
dropout/mulMuldropout/divdropout/Floor*
T0

Conv2D_2Conv2Ddropout/mulVariable_4/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC
0
add_2AddConv2D_2Variable_5/read*
T0

Relu_2Reluadd_2*
T0
w
	MaxPool_1MaxPoolRelu_2*
T0*
ksize
*
strides
*
paddingSAME*
data_formatNHWC
<
dropout_1/ShapeShape	MaxPool_1*
T0*
out_type0
I
dropout_1/random_uniform/minConst*
valueB
 *    *
dtype0
I
dropout_1/random_uniform/maxConst*
valueB
 *  ?*
dtype0
w
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape*

seed *
seed2 *
dtype0*
T0
h
dropout_1/random_uniform/subSubdropout_1/random_uniform/maxdropout_1/random_uniform/min*
T0
r
dropout_1/random_uniform/mulMul&dropout_1/random_uniform/RandomUniformdropout_1/random_uniform/sub*
T0
d
dropout_1/random_uniformAdddropout_1/random_uniform/muldropout_1/random_uniform/min*
T0
F
dropout_1/addAddPlaceholder_1dropout_1/random_uniform*
T0
0
dropout_1/FloorFloordropout_1/add*
T0
;
dropout_1/divRealDiv	MaxPool_1Placeholder_1*
T0
=
dropout_1/mulMuldropout_1/divdropout_1/Floor*
T0
B
Reshape/shapeConst*
valueB"ÿÿÿÿ  *
dtype0
G
ReshapeReshapedropout_1/mulReshape/shape*
T0*
Tshape0
Y
MatMulMatMulReshapeVariable_6/read*
transpose_a( *
transpose_b( *
T0
.
add_3AddMatMulVariable_7/read*
T0

Relu_3Reluadd_3*
T0
9
dropout_2/ShapeShapeRelu_3*
T0*
out_type0
I
dropout_2/random_uniform/minConst*
valueB
 *    *
dtype0
I
dropout_2/random_uniform/maxConst*
valueB
 *  ?*
dtype0
w
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape*

seed *
seed2 *
dtype0*
T0
h
dropout_2/random_uniform/subSubdropout_2/random_uniform/maxdropout_2/random_uniform/min*
T0
r
dropout_2/random_uniform/mulMul&dropout_2/random_uniform/RandomUniformdropout_2/random_uniform/sub*
T0
d
dropout_2/random_uniformAdddropout_2/random_uniform/muldropout_2/random_uniform/min*
T0
F
dropout_2/addAddPlaceholder_1dropout_2/random_uniform*
T0
0
dropout_2/FloorFloordropout_2/add*
T0
8
dropout_2/divRealDivRelu_3Placeholder_1*
T0
=
dropout_2/mulMuldropout_2/divdropout_2/Floor*
T0
Z
MatMul_1MatMulRelu_3Variable_8/read*
transpose_a( *
transpose_b( *
T0
0
add_4AddMatMul_1Variable_9/read*
T0
&
yhat_outputSoftmaxadd_4*
T0
.
RankConst*
value	B :*
dtype0
.
ShapeShapeadd_4*
T0*
out_type0
0
Rank_1Const*
value	B :*
dtype0
0
Shape_1Shapeadd_4*
T0*
out_type0
/
Sub/yConst*
value	B :*
dtype0
"
SubSubRank_1Sub/y*
T0
6
Slice/beginPackSub*
N*
T0*

axis 
8

Slice/sizeConst*
valueB:*
dtype0
F
SliceSliceShape_1Slice/begin
Slice/size*
T0*
Index0
F
concat/values_0Const*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0
5
concat/axisConst*
value	B : *
dtype0
U
concatConcatV2concat/values_0Sliceconcat/axis*
N*
T0*

Tidx0
:
	Reshape_1Reshapeadd_4concat*
T0*
Tshape0
0
Rank_2Const*
value	B :*
dtype0
6
Shape_2ShapePlaceholder*
T0*
out_type0
1
Sub_1/yConst*
value	B :*
dtype0
&
Sub_1SubRank_2Sub_1/y*
T0
:
Slice_1/beginPackSub_1*
N*
T0*

axis 
:
Slice_1/sizeConst*
valueB:*
dtype0
L
Slice_1SliceShape_2Slice_1/beginSlice_1/size*
T0*
Index0
H
concat_1/values_0Const*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0
7
concat_1/axisConst*
value	B : *
dtype0
]
concat_1ConcatV2concat_1/values_0Slice_1concat_1/axis*
N*
T0*

Tidx0
B
	Reshape_2ReshapePlaceholderconcat_1*
T0*
Tshape0
]
SoftmaxCrossEntropyWithLogitsSoftmaxCrossEntropyWithLogits	Reshape_1	Reshape_2*
T0
1
Sub_2/yConst*
value	B :*
dtype0
$
Sub_2SubRankSub_2/y*
T0
;
Slice_2/beginConst*
valueB: *
dtype0
9
Slice_2/sizePackSub_2*
N*
T0*

axis 
J
Slice_2SliceShapeSlice_2/beginSlice_2/size*
T0*
Index0
S
	Reshape_3ReshapeSoftmaxCrossEntropyWithLogitsSlice_2*
T0*
Tshape0
3
ConstConst*
valueB: *
dtype0
D
MeanMean	Reshape_3Const*
	keep_dims( *
T0*

Tidx0
8
gradients/ShapeConst*
valueB *
dtype0
<
gradients/ConstConst*
valueB
 *  ?*
dtype0
A
gradients/FillFillgradients/Shapegradients/Const*
T0
O
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0
p
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0
F
gradients/Mean_grad/ShapeShape	Reshape_3*
T0*
out_type0
s
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*

Tmultiples0
H
gradients/Mean_grad/Shape_1Shape	Reshape_3*
T0*
out_type0
D
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0
w
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
®
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
T0*

Tidx0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
y
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
²
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
T0*

Tidx0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
w
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*.
_class$
" loc:@gradients/Mean_grad/Shape_1

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*.
_class$
" loc:@gradients/Mean_grad/Shape_1
V
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*

DstT0
c
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0
_
gradients/Reshape_3_grad/ShapeShapeSoftmaxCrossEntropyWithLogits*
T0*
out_type0

 gradients/Reshape_3_grad/ReshapeReshapegradients/Mean_grad/truedivgradients/Reshape_3_grad/Shape*
T0*
Tshape0
K
gradients/zeros_like	ZerosLikeSoftmaxCrossEntropyWithLogits:1*
T0
n
;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0
¹
7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDims gradients/Reshape_3_grad/Reshape;gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*

Tdim0

0gradients/SoftmaxCrossEntropyWithLogits_grad/mulMul7gradients/SoftmaxCrossEntropyWithLogits_grad/ExpandDimsSoftmaxCrossEntropyWithLogits:1*
T0
G
gradients/Reshape_1_grad/ShapeShapeadd_4*
T0*
out_type0

 gradients/Reshape_1_grad/ReshapeReshape0gradients/SoftmaxCrossEntropyWithLogits_grad/mulgradients/Reshape_1_grad/Shape*
T0*
Tshape0
F
gradients/add_4_grad/ShapeShapeMatMul_1*
T0*
out_type0
J
gradients/add_4_grad/Shape_1Const*
valueB:
*
dtype0

*gradients/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_4_grad/Shapegradients/add_4_grad/Shape_1*
T0

gradients/add_4_grad/SumSum gradients/Reshape_1_grad/Reshape*gradients/add_4_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0
t
gradients/add_4_grad/ReshapeReshapegradients/add_4_grad/Sumgradients/add_4_grad/Shape*
T0*
Tshape0

gradients/add_4_grad/Sum_1Sum gradients/Reshape_1_grad/Reshape,gradients/add_4_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0
z
gradients/add_4_grad/Reshape_1Reshapegradients/add_4_grad/Sum_1gradients/add_4_grad/Shape_1*
T0*
Tshape0
m
%gradients/add_4_grad/tuple/group_depsNoOp^gradients/add_4_grad/Reshape^gradients/add_4_grad/Reshape_1
¹
-gradients/add_4_grad/tuple/control_dependencyIdentitygradients/add_4_grad/Reshape&^gradients/add_4_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_4_grad/Reshape
¿
/gradients/add_4_grad/tuple/control_dependency_1Identitygradients/add_4_grad/Reshape_1&^gradients/add_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_4_grad/Reshape_1

gradients/MatMul_1_grad/MatMulMatMul-gradients/add_4_grad/tuple/control_dependencyVariable_8/read*
transpose_a( *
transpose_b(*
T0

 gradients/MatMul_1_grad/MatMul_1MatMulRelu_3-gradients/add_4_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
Ã
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul
É
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1
m
gradients/Relu_3_grad/ReluGradReluGrad0gradients/MatMul_1_grad/tuple/control_dependencyRelu_3*
T0
D
gradients/add_3_grad/ShapeShapeMatMul*
T0*
out_type0
K
gradients/add_3_grad/Shape_1Const*
valueB:*
dtype0

*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*
T0

gradients/add_3_grad/SumSumgradients/Relu_3_grad/ReluGrad*gradients/add_3_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0
t
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
T0*
Tshape0

gradients/add_3_grad/Sum_1Sumgradients/Relu_3_grad/ReluGrad,gradients/add_3_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0
z
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
T0*
Tshape0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
¹
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_3_grad/Reshape
¿
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1

gradients/MatMul_grad/MatMulMatMul-gradients/add_3_grad/tuple/control_dependencyVariable_6/read*
transpose_a( *
transpose_b(*
T0

gradients/MatMul_grad/MatMul_1MatMulReshape-gradients/add_3_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
»
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
Á
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
M
gradients/Reshape_grad/ShapeShapedropout_1/mul*
T0*
out_type0

gradients/Reshape_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*
T0*
Tshape0
S
"gradients/dropout_1/mul_grad/ShapeShapedropout_1/div*
T0*
out_type0
W
$gradients/dropout_1/mul_grad/Shape_1Shapedropout_1/Floor*
T0*
out_type0

2gradients/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_1/mul_grad/Shape$gradients/dropout_1/mul_grad/Shape_1*
T0
a
 gradients/dropout_1/mul_grad/mulMulgradients/Reshape_grad/Reshapedropout_1/Floor*
T0
£
 gradients/dropout_1/mul_grad/SumSum gradients/dropout_1/mul_grad/mul2gradients/dropout_1/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0

$gradients/dropout_1/mul_grad/ReshapeReshape gradients/dropout_1/mul_grad/Sum"gradients/dropout_1/mul_grad/Shape*
T0*
Tshape0
a
"gradients/dropout_1/mul_grad/mul_1Muldropout_1/divgradients/Reshape_grad/Reshape*
T0
©
"gradients/dropout_1/mul_grad/Sum_1Sum"gradients/dropout_1/mul_grad/mul_14gradients/dropout_1/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0

&gradients/dropout_1/mul_grad/Reshape_1Reshape"gradients/dropout_1/mul_grad/Sum_1$gradients/dropout_1/mul_grad/Shape_1*
T0*
Tshape0

-gradients/dropout_1/mul_grad/tuple/group_depsNoOp%^gradients/dropout_1/mul_grad/Reshape'^gradients/dropout_1/mul_grad/Reshape_1
Ù
5gradients/dropout_1/mul_grad/tuple/control_dependencyIdentity$gradients/dropout_1/mul_grad/Reshape.^gradients/dropout_1/mul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dropout_1/mul_grad/Reshape
ß
7gradients/dropout_1/mul_grad/tuple/control_dependency_1Identity&gradients/dropout_1/mul_grad/Reshape_1.^gradients/dropout_1/mul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/dropout_1/mul_grad/Reshape_1
O
"gradients/dropout_1/div_grad/ShapeShape	MaxPool_1*
T0*
out_type0
U
$gradients/dropout_1/div_grad/Shape_1ShapePlaceholder_1*
T0*
out_type0

2gradients/dropout_1/div_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/dropout_1/div_grad/Shape$gradients/dropout_1/div_grad/Shape_1*
T0
~
$gradients/dropout_1/div_grad/RealDivRealDiv5gradients/dropout_1/mul_grad/tuple/control_dependencyPlaceholder_1*
T0
§
 gradients/dropout_1/div_grad/SumSum$gradients/dropout_1/div_grad/RealDiv2gradients/dropout_1/div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0

$gradients/dropout_1/div_grad/ReshapeReshape gradients/dropout_1/div_grad/Sum"gradients/dropout_1/div_grad/Shape*
T0*
Tshape0
;
 gradients/dropout_1/div_grad/NegNeg	MaxPool_1*
T0
k
&gradients/dropout_1/div_grad/RealDiv_1RealDiv gradients/dropout_1/div_grad/NegPlaceholder_1*
T0
q
&gradients/dropout_1/div_grad/RealDiv_2RealDiv&gradients/dropout_1/div_grad/RealDiv_1Placeholder_1*
T0

 gradients/dropout_1/div_grad/mulMul5gradients/dropout_1/mul_grad/tuple/control_dependency&gradients/dropout_1/div_grad/RealDiv_2*
T0
§
"gradients/dropout_1/div_grad/Sum_1Sum gradients/dropout_1/div_grad/mul4gradients/dropout_1/div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0

&gradients/dropout_1/div_grad/Reshape_1Reshape"gradients/dropout_1/div_grad/Sum_1$gradients/dropout_1/div_grad/Shape_1*
T0*
Tshape0

-gradients/dropout_1/div_grad/tuple/group_depsNoOp%^gradients/dropout_1/div_grad/Reshape'^gradients/dropout_1/div_grad/Reshape_1
Ù
5gradients/dropout_1/div_grad/tuple/control_dependencyIdentity$gradients/dropout_1/div_grad/Reshape.^gradients/dropout_1/div_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dropout_1/div_grad/Reshape
ß
7gradients/dropout_1/div_grad/tuple/control_dependency_1Identity&gradients/dropout_1/div_grad/Reshape_1.^gradients/dropout_1/div_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/dropout_1/div_grad/Reshape_1
Ø
$gradients/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_15gradients/dropout_1/div_grad/tuple/control_dependency*
ksize
*
strides
*
paddingSAME*
data_formatNHWC*
T0
a
gradients/Relu_2_grad/ReluGradReluGrad$gradients/MaxPool_1_grad/MaxPoolGradRelu_2*
T0
F
gradients/add_2_grad/ShapeShapeConv2D_2*
T0*
out_type0
J
gradients/add_2_grad/Shape_1Const*
valueB:*
dtype0

*gradients/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_2_grad/Shapegradients/add_2_grad/Shape_1*
T0

gradients/add_2_grad/SumSumgradients/Relu_2_grad/ReluGrad*gradients/add_2_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0
t
gradients/add_2_grad/ReshapeReshapegradients/add_2_grad/Sumgradients/add_2_grad/Shape*
T0*
Tshape0

gradients/add_2_grad/Sum_1Sumgradients/Relu_2_grad/ReluGrad,gradients/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0
z
gradients/add_2_grad/Reshape_1Reshapegradients/add_2_grad/Sum_1gradients/add_2_grad/Shape_1*
T0*
Tshape0
m
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/add_2_grad/Reshape^gradients/add_2_grad/Reshape_1
¹
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/add_2_grad/Reshape&^gradients/add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_2_grad/Reshape
¿
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/add_2_grad/Reshape_1&^gradients/add_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_2_grad/Reshape_1
h
gradients/Conv2D_2_grad/ShapeNShapeNdropout/mulVariable_4/read*
N*
T0*
out_type0

+gradients/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_2_grad/ShapeNVariable_4/read-gradients/add_2_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC

,gradients/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFilterdropout/mul gradients/Conv2D_2_grad/ShapeN:1-gradients/add_2_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC

(gradients/Conv2D_2_grad/tuple/group_depsNoOp,^gradients/Conv2D_2_grad/Conv2DBackpropInput-^gradients/Conv2D_2_grad/Conv2DBackpropFilter
Ý
0gradients/Conv2D_2_grad/tuple/control_dependencyIdentity+gradients/Conv2D_2_grad/Conv2DBackpropInput)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_2_grad/Conv2DBackpropInput
á
2gradients/Conv2D_2_grad/tuple/control_dependency_1Identity,gradients/Conv2D_2_grad/Conv2DBackpropFilter)^gradients/Conv2D_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_2_grad/Conv2DBackpropFilter
O
 gradients/dropout/mul_grad/ShapeShapedropout/div*
T0*
out_type0
S
"gradients/dropout/mul_grad/Shape_1Shapedropout/Floor*
T0*
out_type0

0gradients/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/mul_grad/Shape"gradients/dropout/mul_grad/Shape_1*
T0
o
gradients/dropout/mul_grad/mulMul0gradients/Conv2D_2_grad/tuple/control_dependencydropout/Floor*
T0

gradients/dropout/mul_grad/SumSumgradients/dropout/mul_grad/mul0gradients/dropout/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0

"gradients/dropout/mul_grad/ReshapeReshapegradients/dropout/mul_grad/Sum gradients/dropout/mul_grad/Shape*
T0*
Tshape0
o
 gradients/dropout/mul_grad/mul_1Muldropout/div0gradients/Conv2D_2_grad/tuple/control_dependency*
T0
£
 gradients/dropout/mul_grad/Sum_1Sum gradients/dropout/mul_grad/mul_12gradients/dropout/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0

$gradients/dropout/mul_grad/Reshape_1Reshape gradients/dropout/mul_grad/Sum_1"gradients/dropout/mul_grad/Shape_1*
T0*
Tshape0

+gradients/dropout/mul_grad/tuple/group_depsNoOp#^gradients/dropout/mul_grad/Reshape%^gradients/dropout/mul_grad/Reshape_1
Ñ
3gradients/dropout/mul_grad/tuple/control_dependencyIdentity"gradients/dropout/mul_grad/Reshape,^gradients/dropout/mul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dropout/mul_grad/Reshape
×
5gradients/dropout/mul_grad/tuple/control_dependency_1Identity$gradients/dropout/mul_grad/Reshape_1,^gradients/dropout/mul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dropout/mul_grad/Reshape_1
K
 gradients/dropout/div_grad/ShapeShapeMaxPool*
T0*
out_type0
S
"gradients/dropout/div_grad/Shape_1ShapePlaceholder_1*
T0*
out_type0

0gradients/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs gradients/dropout/div_grad/Shape"gradients/dropout/div_grad/Shape_1*
T0
z
"gradients/dropout/div_grad/RealDivRealDiv3gradients/dropout/mul_grad/tuple/control_dependencyPlaceholder_1*
T0
¡
gradients/dropout/div_grad/SumSum"gradients/dropout/div_grad/RealDiv0gradients/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0

"gradients/dropout/div_grad/ReshapeReshapegradients/dropout/div_grad/Sum gradients/dropout/div_grad/Shape*
T0*
Tshape0
7
gradients/dropout/div_grad/NegNegMaxPool*
T0
g
$gradients/dropout/div_grad/RealDiv_1RealDivgradients/dropout/div_grad/NegPlaceholder_1*
T0
m
$gradients/dropout/div_grad/RealDiv_2RealDiv$gradients/dropout/div_grad/RealDiv_1Placeholder_1*
T0

gradients/dropout/div_grad/mulMul3gradients/dropout/mul_grad/tuple/control_dependency$gradients/dropout/div_grad/RealDiv_2*
T0
¡
 gradients/dropout/div_grad/Sum_1Sumgradients/dropout/div_grad/mul2gradients/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0

$gradients/dropout/div_grad/Reshape_1Reshape gradients/dropout/div_grad/Sum_1"gradients/dropout/div_grad/Shape_1*
T0*
Tshape0

+gradients/dropout/div_grad/tuple/group_depsNoOp#^gradients/dropout/div_grad/Reshape%^gradients/dropout/div_grad/Reshape_1
Ñ
3gradients/dropout/div_grad/tuple/control_dependencyIdentity"gradients/dropout/div_grad/Reshape,^gradients/dropout/div_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dropout/div_grad/Reshape
×
5gradients/dropout/div_grad/tuple/control_dependency_1Identity$gradients/dropout/div_grad/Reshape_1,^gradients/dropout/div_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dropout/div_grad/Reshape_1
Ò
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradRelu_1MaxPool3gradients/dropout/div_grad/tuple/control_dependency*
ksize
*
strides
*
paddingSAME*
data_formatNHWC*
T0
_
gradients/Relu_1_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu_1*
T0
F
gradients/add_1_grad/ShapeShapeConv2D_1*
T0*
out_type0
J
gradients/add_1_grad/Shape_1Const*
valueB:*
dtype0

*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0

gradients/add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/add_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0
t
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
T0*
Tshape0

gradients/add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0
z
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
T0*
Tshape0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
¹
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape
¿
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1
a
gradients/Conv2D_1_grad/ShapeNShapeNReluVariable_2/read*
N*
T0*
out_type0

+gradients/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_1_grad/ShapeNVariable_2/read-gradients/add_1_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC
û
,gradients/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFilterRelu gradients/Conv2D_1_grad/ShapeN:1-gradients/add_1_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC

(gradients/Conv2D_1_grad/tuple/group_depsNoOp,^gradients/Conv2D_1_grad/Conv2DBackpropInput-^gradients/Conv2D_1_grad/Conv2DBackpropFilter
Ý
0gradients/Conv2D_1_grad/tuple/control_dependencyIdentity+gradients/Conv2D_1_grad/Conv2DBackpropInput)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*>
_class4
20loc:@gradients/Conv2D_1_grad/Conv2DBackpropInput
á
2gradients/Conv2D_1_grad/tuple/control_dependency_1Identity,gradients/Conv2D_1_grad/Conv2DBackpropFilter)^gradients/Conv2D_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/Conv2D_1_grad/Conv2DBackpropFilter
i
gradients/Relu_grad/ReluGradReluGrad0gradients/Conv2D_1_grad/tuple/control_dependencyRelu*
T0
B
gradients/add_grad/ShapeShapeConv2D*
T0*
out_type0
H
gradients/add_grad/Shape_1Const*
valueB:*
dtype0

(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0

gradients/add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0
n
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0

gradients/add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0
t
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
±
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape
·
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
`
gradients/Conv2D_grad/ShapeNShapeNX_inputVariable/read*
N*
T0*
out_type0
ú
)gradients/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInputgradients/Conv2D_grad/ShapeNVariable/read+gradients/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC
ø
*gradients/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterX_inputgradients/Conv2D_grad/ShapeN:1+gradients/add_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC

&gradients/Conv2D_grad/tuple/group_depsNoOp*^gradients/Conv2D_grad/Conv2DBackpropInput+^gradients/Conv2D_grad/Conv2DBackpropFilter
Õ
.gradients/Conv2D_grad/tuple/control_dependencyIdentity)gradients/Conv2D_grad/Conv2DBackpropInput'^gradients/Conv2D_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/Conv2D_grad/Conv2DBackpropInput
Ù
0gradients/Conv2D_grad/tuple/control_dependency_1Identity*gradients/Conv2D_grad/Conv2DBackpropFilter'^gradients/Conv2D_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/Conv2D_grad/Conv2DBackpropFilter
c
beta1_power/initial_valueConst*
valueB
 *fff?*
dtype0*
_class
loc:@Variable
t
beta1_power
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_class
loc:@Variable

beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable
O
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@Variable
c
beta2_power/initial_valueConst*
valueB
 *w¾?*
dtype0*
_class
loc:@Variable
t
beta2_power
VariableV2*
shape: *
dtype0*
	container *
shared_name *
_class
loc:@Variable

beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable
O
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@Variable
y
Variable/Adam/Initializer/zerosConst*%
valueB*    *
dtype0*
_class
loc:@Variable

Variable/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@Variable

Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable
S
Variable/Adam/readIdentityVariable/Adam*
T0*
_class
loc:@Variable
{
!Variable/Adam_1/Initializer/zerosConst*%
valueB*    *
dtype0*
_class
loc:@Variable

Variable/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@Variable
£
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable
W
Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable
q
!Variable_1/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@Variable_1
~
Variable_1/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@Variable_1
¥
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_1
Y
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_class
loc:@Variable_1
s
#Variable_1/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@Variable_1

Variable_1/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@Variable_1
«
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_1
]
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_class
loc:@Variable_1
}
!Variable_2/Adam/Initializer/zerosConst*%
valueB*    *
dtype0*
_class
loc:@Variable_2

Variable_2/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@Variable_2
¥
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_2
Y
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_class
loc:@Variable_2

#Variable_2/Adam_1/Initializer/zerosConst*%
valueB*    *
dtype0*
_class
loc:@Variable_2

Variable_2/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@Variable_2
«
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_2
]
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2
q
!Variable_3/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@Variable_3
~
Variable_3/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@Variable_3
¥
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_3
Y
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3
s
#Variable_3/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@Variable_3

Variable_3/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@Variable_3
«
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_3
]
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3
}
!Variable_4/Adam/Initializer/zerosConst*%
valueB*    *
dtype0*
_class
loc:@Variable_4

Variable_4/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@Variable_4
¥
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_4
Y
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*
_class
loc:@Variable_4

#Variable_4/Adam_1/Initializer/zerosConst*%
valueB*    *
dtype0*
_class
loc:@Variable_4

Variable_4/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@Variable_4
«
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_4
]
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*
_class
loc:@Variable_4
q
!Variable_5/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@Variable_5
~
Variable_5/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@Variable_5
¥
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_5
Y
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_class
loc:@Variable_5
s
#Variable_5/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@Variable_5

Variable_5/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@Variable_5
«
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_5
]
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_class
loc:@Variable_5
w
!Variable_6/Adam/Initializer/zerosConst*
valueB
*    *
dtype0*
_class
loc:@Variable_6

Variable_6/Adam
VariableV2*
shape:
*
dtype0*
	container *
shared_name *
_class
loc:@Variable_6
¥
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_6
Y
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6
y
#Variable_6/Adam_1/Initializer/zerosConst*
valueB
*    *
dtype0*
_class
loc:@Variable_6

Variable_6/Adam_1
VariableV2*
shape:
*
dtype0*
	container *
shared_name *
_class
loc:@Variable_6
«
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_6
]
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6
r
!Variable_7/Adam/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@Variable_7

Variable_7/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@Variable_7
¥
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_7
Y
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_class
loc:@Variable_7
t
#Variable_7/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*
_class
loc:@Variable_7

Variable_7/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *
_class
loc:@Variable_7
«
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_7
]
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_class
loc:@Variable_7
v
!Variable_8/Adam/Initializer/zerosConst*
valueB	
*    *
dtype0*
_class
loc:@Variable_8

Variable_8/Adam
VariableV2*
shape:	
*
dtype0*
	container *
shared_name *
_class
loc:@Variable_8
¥
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_8
Y
Variable_8/Adam/readIdentityVariable_8/Adam*
T0*
_class
loc:@Variable_8
x
#Variable_8/Adam_1/Initializer/zerosConst*
valueB	
*    *
dtype0*
_class
loc:@Variable_8

Variable_8/Adam_1
VariableV2*
shape:	
*
dtype0*
	container *
shared_name *
_class
loc:@Variable_8
«
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_8
]
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
T0*
_class
loc:@Variable_8
q
!Variable_9/Adam/Initializer/zerosConst*
valueB
*    *
dtype0*
_class
loc:@Variable_9
~
Variable_9/Adam
VariableV2*
shape:
*
dtype0*
	container *
shared_name *
_class
loc:@Variable_9
¥
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_9
Y
Variable_9/Adam/readIdentityVariable_9/Adam*
T0*
_class
loc:@Variable_9
s
#Variable_9/Adam_1/Initializer/zerosConst*
valueB
*    *
dtype0*
_class
loc:@Variable_9

Variable_9/Adam_1
VariableV2*
shape:
*
dtype0*
	container *
shared_name *
_class
loc:@Variable_9
«
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_9
]
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
T0*
_class
loc:@Variable_9
?
Adam/learning_rateConst*
valueB
 *o:*
dtype0
7

Adam/beta1Const*
valueB
 *fff?*
dtype0
7

Adam/beta2Const*
valueB
 *w¾?*
dtype0
9
Adam/epsilonConst*
valueB
 *wÌ+2*
dtype0
²
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/Conv2D_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_class
loc:@Variable
¹
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_class
loc:@Variable_1
¾
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_class
loc:@Variable_2
»
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_class
loc:@Variable_3
¾
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/Conv2D_2_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_class
loc:@Variable_4
»
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_2_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_class
loc:@Variable_5
¼
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_class
loc:@Variable_6
»
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_3_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_class
loc:@Variable_7
¾
 Adam/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_class
loc:@Variable_8
»
 Adam/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/add_4_grad/tuple/control_dependency_1*
T0*
use_locking( *
use_nesterov( *
_class
loc:@Variable_9
¯
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam*
T0*
_class
loc:@Variable
{
Adam/AssignAssignbeta1_powerAdam/mul*
T0*
validate_shape(*
use_locking( *
_class
loc:@Variable
±

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam*
T0*
_class
loc:@Variable

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
validate_shape(*
use_locking( *
_class
loc:@Variable

AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam!^Adam/update_Variable_8/ApplyAdam!^Adam/update_Variable_9/ApplyAdam^Adam/Assign^Adam/Assign_1
:
ArgMax/dimensionConst*
value	B :*
dtype0
W
ArgMaxArgMaxPlaceholderArgMax/dimension*
T0*

Tidx0*
output_type0	
<
ArgMax_1/dimensionConst*
value	B :*
dtype0
[
ArgMax_1ArgMaxyhat_outputArgMax_1/dimension*
T0*

Tidx0*
output_type0	
)
EqualEqualArgMaxArgMax_1*
T0	
-
Cast_1CastEqual*

SrcT0
*

DstT0
5
Const_1Const*
valueB: *
dtype0
E
Mean_1MeanCast_1Const_1*
	keep_dims( *
T0*

Tidx0

initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^Variable_8/Assign^Variable_9/Assign^beta1_power/Assign^beta2_power/Assign^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign^Variable_8/Adam/Assign^Variable_8/Adam_1/Assign^Variable_9/Adam/Assign^Variable_9/Adam_1/Assign
8

save/ConstConst*
valueB Bmodel*
dtype0
º
save/SaveV2/tensor_namesConst*
valueÿBü BVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6BVariable_6/AdamBVariable_6/Adam_1B
Variable_7BVariable_7/AdamBVariable_7/Adam_1B
Variable_8BVariable_8/AdamBVariable_8/Adam_1B
Variable_9BVariable_9/AdamBVariable_9/Adam_1Bbeta1_powerBbeta2_power*
dtype0

save/SaveV2/shape_and_slicesConst*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ý
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariableVariable/AdamVariable/Adam_1
Variable_1Variable_1/AdamVariable_1/Adam_1
Variable_2Variable_2/AdamVariable_2/Adam_1
Variable_3Variable_3/AdamVariable_3/Adam_1
Variable_4Variable_4/AdamVariable_4/Adam_1
Variable_5Variable_5/AdamVariable_5/Adam_1
Variable_6Variable_6/AdamVariable_6/Adam_1
Variable_7Variable_7/AdamVariable_7/Adam_1
Variable_8Variable_8/AdamVariable_8/Adam_1
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_powerbeta2_power*.
dtypes$
"2 
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const
P
save/RestoreV2/tensor_namesConst*
valueBBVariable*
dtype0
L
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0
v
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2
~
save/AssignAssignVariablesave/RestoreV2*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable
W
save/RestoreV2_1/tensor_namesConst*"
valueBBVariable/Adam*
dtype0
N
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2

save/Assign_1AssignVariable/Adamsave/RestoreV2_1*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable
Y
save/RestoreV2_2/tensor_namesConst*$
valueBBVariable/Adam_1*
dtype0
N
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2

save/Assign_2AssignVariable/Adam_1save/RestoreV2_2*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable
T
save/RestoreV2_3/tensor_namesConst*
valueBB
Variable_1*
dtype0
N
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2

save/Assign_3Assign
Variable_1save/RestoreV2_3*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_1
Y
save/RestoreV2_4/tensor_namesConst*$
valueBBVariable_1/Adam*
dtype0
N
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2

save/Assign_4AssignVariable_1/Adamsave/RestoreV2_4*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_1
[
save/RestoreV2_5/tensor_namesConst*&
valueBBVariable_1/Adam_1*
dtype0
N
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2

save/Assign_5AssignVariable_1/Adam_1save/RestoreV2_5*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_1
T
save/RestoreV2_6/tensor_namesConst*
valueBB
Variable_2*
dtype0
N
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2

save/Assign_6Assign
Variable_2save/RestoreV2_6*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_2
Y
save/RestoreV2_7/tensor_namesConst*$
valueBBVariable_2/Adam*
dtype0
N
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2

save/Assign_7AssignVariable_2/Adamsave/RestoreV2_7*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_2
[
save/RestoreV2_8/tensor_namesConst*&
valueBBVariable_2/Adam_1*
dtype0
N
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2

save/Assign_8AssignVariable_2/Adam_1save/RestoreV2_8*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_2
T
save/RestoreV2_9/tensor_namesConst*
valueBB
Variable_3*
dtype0
N
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2

save/Assign_9Assign
Variable_3save/RestoreV2_9*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_3
Z
save/RestoreV2_10/tensor_namesConst*$
valueBBVariable_3/Adam*
dtype0
O
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2

save/Assign_10AssignVariable_3/Adamsave/RestoreV2_10*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_3
\
save/RestoreV2_11/tensor_namesConst*&
valueBBVariable_3/Adam_1*
dtype0
O
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2

save/Assign_11AssignVariable_3/Adam_1save/RestoreV2_11*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_3
U
save/RestoreV2_12/tensor_namesConst*
valueBB
Variable_4*
dtype0
O
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2

save/Assign_12Assign
Variable_4save/RestoreV2_12*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_4
Z
save/RestoreV2_13/tensor_namesConst*$
valueBBVariable_4/Adam*
dtype0
O
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2

save/Assign_13AssignVariable_4/Adamsave/RestoreV2_13*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_4
\
save/RestoreV2_14/tensor_namesConst*&
valueBBVariable_4/Adam_1*
dtype0
O
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2

save/Assign_14AssignVariable_4/Adam_1save/RestoreV2_14*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_4
U
save/RestoreV2_15/tensor_namesConst*
valueBB
Variable_5*
dtype0
O
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2

save/Assign_15Assign
Variable_5save/RestoreV2_15*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_5
Z
save/RestoreV2_16/tensor_namesConst*$
valueBBVariable_5/Adam*
dtype0
O
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2

save/Assign_16AssignVariable_5/Adamsave/RestoreV2_16*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_5
\
save/RestoreV2_17/tensor_namesConst*&
valueBBVariable_5/Adam_1*
dtype0
O
"save/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2

save/Assign_17AssignVariable_5/Adam_1save/RestoreV2_17*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_5
U
save/RestoreV2_18/tensor_namesConst*
valueBB
Variable_6*
dtype0
O
"save/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2

save/Assign_18Assign
Variable_6save/RestoreV2_18*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_6
Z
save/RestoreV2_19/tensor_namesConst*$
valueBBVariable_6/Adam*
dtype0
O
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2

save/Assign_19AssignVariable_6/Adamsave/RestoreV2_19*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_6
\
save/RestoreV2_20/tensor_namesConst*&
valueBBVariable_6/Adam_1*
dtype0
O
"save/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2

save/Assign_20AssignVariable_6/Adam_1save/RestoreV2_20*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_6
U
save/RestoreV2_21/tensor_namesConst*
valueBB
Variable_7*
dtype0
O
"save/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2

save/Assign_21Assign
Variable_7save/RestoreV2_21*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_7
Z
save/RestoreV2_22/tensor_namesConst*$
valueBBVariable_7/Adam*
dtype0
O
"save/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2

save/Assign_22AssignVariable_7/Adamsave/RestoreV2_22*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_7
\
save/RestoreV2_23/tensor_namesConst*&
valueBBVariable_7/Adam_1*
dtype0
O
"save/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2

save/Assign_23AssignVariable_7/Adam_1save/RestoreV2_23*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_7
U
save/RestoreV2_24/tensor_namesConst*
valueBB
Variable_8*
dtype0
O
"save/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2

save/Assign_24Assign
Variable_8save/RestoreV2_24*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_8
Z
save/RestoreV2_25/tensor_namesConst*$
valueBBVariable_8/Adam*
dtype0
O
"save/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2

save/Assign_25AssignVariable_8/Adamsave/RestoreV2_25*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_8
\
save/RestoreV2_26/tensor_namesConst*&
valueBBVariable_8/Adam_1*
dtype0
O
"save/RestoreV2_26/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2

save/Assign_26AssignVariable_8/Adam_1save/RestoreV2_26*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_8
U
save/RestoreV2_27/tensor_namesConst*
valueBB
Variable_9*
dtype0
O
"save/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2

save/Assign_27Assign
Variable_9save/RestoreV2_27*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_9
Z
save/RestoreV2_28/tensor_namesConst*$
valueBBVariable_9/Adam*
dtype0
O
"save/RestoreV2_28/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
dtypes
2

save/Assign_28AssignVariable_9/Adamsave/RestoreV2_28*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_9
\
save/RestoreV2_29/tensor_namesConst*&
valueBBVariable_9/Adam_1*
dtype0
O
"save/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2

save/Assign_29AssignVariable_9/Adam_1save/RestoreV2_29*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable_9
V
save/RestoreV2_30/tensor_namesConst* 
valueBBbeta1_power*
dtype0
O
"save/RestoreV2_30/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2

save/Assign_30Assignbeta1_powersave/RestoreV2_30*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable
V
save/RestoreV2_31/tensor_namesConst* 
valueBBbeta2_power*
dtype0
O
"save/RestoreV2_31/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2

save/Assign_31Assignbeta2_powersave/RestoreV2_31*
T0*
validate_shape(*
use_locking(*
_class
loc:@Variable
¬
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31"