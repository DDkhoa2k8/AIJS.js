var LIB = {
    gpuMode:false
}
///
var GPU = {
	setupGPU:(fragment,set,w,h) => {
		//get webGL context
		var data = set;
		const canvas = document.createElement("canvas");
		canvas.width = w;
		canvas.height = h;
		const gl = canvas.getContext("webgl");
		var fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
		gl.shaderSource(fragmentShader, fragment);
		gl.compileShader(fragmentShader);
		var vertexShader = gl.createShader(gl.VERTEX_SHADER);
		gl.shaderSource(vertexShader,`
				attribute vec4 position;
				void main() {
					gl_Position = position;
				}`);
		gl.compileShader(vertexShader);
		const program = gl.createProgram();
		gl.attachShader(program, vertexShader);
		gl.attachShader(program, fragmentShader);
		gl.linkProgram(program);
		gl.useProgram(program);
		return (value) => {
			var seti,texi = 0;
			for (let i = 0,len = set.length;i < len;i++) {
				seti = data[i];
				if (seti.type == "tex") {
					var v = value[i];
					gl.uniform1i(gl.getUniformLocation(program, seti.name), texi);
					gl.activeTexture(gl.TEXTURE0 + texi);
					gl.bindTexture(gl.TEXTURE_2D,(() => {
						var tex = gl.createTexture();
						gl.bindTexture(gl.TEXTURE_2D, tex);
						gl.texImage2D(
							gl.TEXTURE_2D,
							0,
							gl[seti.i],
							v[1],
							v[2],
							0,
							gl[seti.f],
							gl[seti.t],
							v[0],
							null
						);
						gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
						gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
						gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
						gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
						return tex;
					})());
					texi++;
				} else {
					gl["uniform" + seti.type + "v"](gl.getUniformLocation(program, seti.name), value[i]);
				}
			}
			let attr = gl.getAttribLocation(program, "position");
			let buffer = gl.createBuffer();
			gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
			gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
			  -1, -1,
			   1, -1,
			  -1,  1,
			  -1,  1,
			   1, -1,
			   1,  1,
			]), gl.STATIC_DRAW);
			gl.enableVertexAttribArray(attr);
			gl.vertexAttribPointer(
				attr,
				2,
				gl.FLOAT,
				false,
				0,
				0,
			);
			gl.drawArrays(gl.TRIANGLES, 0, 6);
			var output = new Uint8Array((w * h) * 4);
			gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, output);
			return output;
		}
	}
}
var reLU = (input) => {
	if (input < 0) {
		return 0;
	} else {
		return input;
	}
};
var linear = (input,m,b) => {
	return input*m + b;
}
var bin = (input) => {
	if (input < 0) {
		return 0;
	} else {
		return 1;
	}
}
var tanh = (input) => {
	return Math.tanh(input);
}
var sig  = (input) => {
	return 1 / 1 + (Math.E ** -input);
}
var leakyReLU = (input) => {
	if (input >= 0) {
		return input;
	} else {
		return input * 0.01;
	}
}
var paraReLU = (input,a) => {
	if (input >= 0) {
		return input;
	} else {
		return input * a;
	}
}
var ELU = (input,a) => {
	if (input >= 0) {
		return input;
	} else {
		return a * ((Math.E ** input) - 1);
	}
}
///func
function checkWebGL() {
    sup = false;
    type = 0;
    let gl1 = document.createElement('canvas').getContext('webgl');
    if (gl1) {
        sup = true;
        type = 1;
    }
    let gl2 = document.createElement('canvas').getContext('webgl2');
    if (gl2) {
        sup = true;
        type += 2;
    }
    if (type == 0) {
        alert("your browser not suport for webGL or webGL are disable");
    } else if (type == 1) {
        alert("your browser not suport for webGL2 but suport webGL1");
    } else if (type == 3) {
        alert("your browser suport for webGL1 and webGL2");
    } else {
        alert("your browser only suport for webGL2");
    }
    return [sup,type];
}
function weiGen(lt) {
    let wei = [];
    for (let i = 0;i < lt;i++) {
        wei.push(Math.random());
    }
    return wei;
}
///
function ANN(lr,mo,act) {
	this.s = 0;
	this.wGS;
    this.weight = [];
	this.nodeData = [];
	this.netData = [];
	this.propageGPU;
	this.actGPU;
	this.canvas = [];
	if (LIB.gpuMode) {
		this.wGS = checkWebGL();
		if (this.wGS[0]) {
			this.s = this.wGS[1];
		}
	}
	this.PropageWebGL1 = () => {
		let canvas = document.createElement("canvas");
		let gl = canvas.getContext("webgl");
		var ver = `
			attribute vec4 position;
			void main() {
				gl_Position = position;
			}`,
		frag = `
			precision highp float;
			precision highp int;
			vec4 encode_float(float v) {
				if (v == 0.0) { 
					return vec4(0.0,0.0,0.0,0.0);
				} 
				float a = abs(v),
				exp = floor(log2(a)),
				mant = (a * pow(2.0,23.0 - exp)),
				mant1 = floor(mant / 256.0 / 256.0),
				mant2 = mod(floor(mant / 256.0),256.0),
				mant3 = mod(mant,256.0),
				sign = 128.0-128.0 * (a / v),
				e = (sign + exp + 127.0) / 510.0,
				m1 = (mant1 - (128.0 * (1.0 - mod(exp + 127.0,2.0)))) / 255.0,
				m2 = (mant2) / 255.0,
				m3 = (mant3 + 0.5) / 255.0;
				return vec4(m3,m2,m1,e); 
			}
			float decode_float(vec4 v) {
				vec4 bits = v * 255.0;
				float sign = mix(-1.0, 1.0, step(bits[3], 128.0)),
				expo = floor(mod(bits[3] + 0.1, 128.0)) * 2.0 + floor((bits[2] + 0.1) / 128.0) - 127.0,
				sig = bits[0] + bits[1] * 256.0 + floor(mod(bits[2] + 0.1, 128.0)) * 256.0 * 256.0;
				return sign * (1.0 + sig / 8388607.0) * pow(2.0, expo);
			}
			void main() {
				
			}
			gl_FragColor = encode_float(sum);
		}`;
		var fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
		gl.shaderSource(fragmentShader, frag);
		gl.compileShader(fragmentShader);
		var vertexShader = gl.createShader(gl.VERTEX_SHADER);
		gl.shaderSource(vertexShader, ver);
		gl.compileShader(vertexShader);
		program = gl.createProgram();
		gl.attachShader(program, vertexShader);
		gl.attachShader(program, fragmentShader);
		gl.linkProgram(program);
		gl.useProgram(program);
		return [gl,canvas];
	}
	this.PropageWebGL2 = () => {
		
	}
    this.setup = (build) => {
        //s loop
		let wei;
        for (let l = 0,bl = build.length - 1,weiL = [];l < bl;l++) {
			weiL = [];
            for (let i = 0,lc = build[l + 1];i < lc;i++) {
				wei = weiGen(build[l]);
                weiL.push(wei);
            }
            this.weight.push(weiL);
        }
		return this.weight;
        //end loop
    }
	this.Node = (input,weight) => {
		let output = 0;
		for (let i = 0,len = weight.length;i < len;i++) {
			output += input[i] * weight[i];
		}
		return output;
	}
	this.Perceptron = (input,weight,act,a) => {
		let output = []
		,net = []
		,actFunc = window[act];
		for (let n = 0,len = weight.length;n < len;n++) {
			net.push(this.Node(input,weight[n]));
			output.push(actFunc(net[n],a));
		}
		this.netData.push(net);
		this.nodeData.push(output);
		return output;
	}
	this.getContextWebGL1 = (act) => {
		this.propageGPU = this.PropageWebGL1();
		this.actGPU = window[act + "WebGL1"]();
	}
	this.getContextWebGL2 = (act) => {
		this.propageGPU = this.PropageWebGL2();
		this.actGPU = window[act + "WebGL2"]();
	}
    this.propagation = (input) => {
		var webGL;
		if (this.s == 0) {
			let netIn = input;
			for (let l = 0,len = this.weight.length;l < len;l++) {
				netIn = this.Perceptron(netIn,this.weight[l],"reLU",0);
			}
			return  netIn;
		} else if (this.s == 3) {
			this.getContextWebGL1(act);
		} else {
			this.getContextWebGL2(act);
		}
		return (input) => {
			var _that = this;
		};
    }
	this.backpropagation = (out,pOut) => {

	}
}
