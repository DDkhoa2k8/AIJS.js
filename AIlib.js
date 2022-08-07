
var LIB = {
    gpuMode:false
}
const lE = (function machineIsLittleEndian() {
	const uint8Array = new Uint8Array([0xAA, 0xBB]);
	const uint16array = new Uint16Array(uint8Array.buffer);
	return uint16array[0] === 0xBBAA;
})();
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
//act func
var actList = ["reLU","linear","bin","tanh","sig","leakyReLU","paraReLU","ELU"];
//reLU
var reLU = (input) => {
	if (input < 0) {
		return 0;
	} else {
		return input;
	}
};
var reLUWebGL1 = () => {
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
		uniform sampler2D net_input;
		uniform bool lE;
		uniform vec2 srcD;
		//float to rgba
		float shiftRight (float v, float amt) {
			v = floor(v) + 0.5;
			return floor(v / exp2(amt));
		}
		float shiftLeft (float v, float amt) {
			return floor(v * exp2(amt) + 0.5);
		}
		float maskLast (float v, float bits) {
			return mod(v, shiftLeft(1.0, bits));
		}
		float extractBits (float num, float from, float to) {
			from = floor(from + 0.5); to = floor(to + 0.5);
			return maskLast(shiftRight(num, from), to - from);
		}
		vec4 floatToRgba(float texelFloat, bool littleEndian) {
			if (texelFloat == 0.0) return vec4(0, 0, 0, 0);
			float sign = texelFloat > 0.0 ? 0.0 : 1.0;
			texelFloat = abs(texelFloat);
			float exponent = floor(log2(texelFloat));
			float biased_exponent = exponent + 127.0;
			float fraction = ((texelFloat / exp2(exponent)) - 1.0) * 8388608.0;
			float t = biased_exponent / 2.0;
			float last_bit_of_biased_exponent = fract(t) * 2.0;
			float remaining_bits_of_biased_exponent = floor(t);
			float byte4 = extractBits(fraction, 0.0, 8.0) / 255.0;
			float byte3 = extractBits(fraction, 8.0, 16.0) / 255.0;
			float byte2 = (last_bit_of_biased_exponent * 128.0 + extractBits(fraction, 16.0, 23.0)) / 255.0;
			float byte1 = (sign * 128.0 + remaining_bits_of_biased_exponent) / 255.0;
			return (
				littleEndian
				? vec4(byte4, byte3, byte2, byte1)
				: vec4(byte1, byte2, byte3, byte4)
			);
		}
		//rgba to float
		ivec4 floatsToBytes(vec4 inputFloats, bool littleEndian) {
			ivec4 bytes = ivec4(inputFloats * 255.0);
			return (
				littleEndian
				? bytes.abgr
				: bytes
			);
		}
		void bytesToBits(const in ivec4 bytes, out bool bits[32]) {
			for (int channelIndex = 0; channelIndex < 4; ++channelIndex) {
				  float acc = float(bytes[channelIndex]);
				  for (int indexInByte = 7; indexInByte >= 0; --indexInByte) {
					float powerOfTwo = exp2(float(indexInByte));
					bool bit = acc >= powerOfTwo;
					bits[channelIndex * 8 + (7 - indexInByte)] = bit;
					acc = mod(acc, powerOfTwo);
				  }
			}
		}
		float getExponent(bool bits[32]) {
			const int startIndex = 1;
			const int bitStringLength = 8;
			const int endBeforeIndex = startIndex + bitStringLength;
			float acc = 0.0;
			int pow2 = bitStringLength - 1;
			for (int bitIndex = startIndex; bitIndex < endBeforeIndex; ++bitIndex) {
				  acc += float(bits[bitIndex]) * exp2(float(pow2--));
			}
			return acc;
		}
		float getMantissa(bool bits[32], bool subnormal) {
			const int startIndex = 9;
			const int bitStringLength = 23;
			const int endBeforeIndex = startIndex + bitStringLength;
			// Leading/implicit/hidden bit convention:
			// If the number is not subnormal (with exponent 0), we add a leading 1 digit.
			float acc = float(!subnormal) * exp2(float(bitStringLength));
			int pow2 = bitStringLength - 1;
			for (int bitIndex = startIndex; bitIndex < endBeforeIndex; ++bitIndex) {
				  acc += float(bits[bitIndex]) * exp2(float(pow2--));
			}
			return acc;
		}
		float bitsToFloat(bool bits[32]) {
			float signBit = float(bits[0]) * -2.0 + 1.0;
			float exponent = getExponent(bits);
			bool subnormal = abs(exponent - 0.0) < 0.01;
			float mantissa = getMantissa(bits, subnormal);
			float exponentBias = 127.0;
			return signBit * mantissa * exp2(exponent - exponentBias - 23.0);
		}
		float rgbaToFloat(vec4 texelRGBA, bool littleEndian) {
			ivec4 rgbaBytes = floatsToBytes(texelRGBA, littleEndian);
			bool bits[32];
			bytesToBits(rgbaBytes, bits);
			return bitsToFloat(bits);
		}
		float node_input
		,result;
		void main() {
			vec2 texcoord = gl_FragCoord.xy / srcD;
			node_input = rgbaToFloat(texture2D(net_input,vec2(texcoord.x,texcoord.y)),lE);
			if (node_input < 0.0) {
				result = 0.0;
			} else {
				result = node_input;
			}
			gl_FragColor = floatToRgba(result,lE);
		}`;
	var fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
	gl.shaderSource(fragmentShader, frag);
	gl.compileShader(fragmentShader);
	var vertexShader = gl.createShader(gl.VERTEX_SHADER);
	gl.shaderSource(vertexShader, ver);
	gl.compileShader(vertexShader);
	var program = gl.createProgram();
	gl.attachShader(program, vertexShader);
	gl.attachShader(program, fragmentShader);
	gl.linkProgram(program);
	gl.useProgram(program);
	return [gl,program,canvas];
}
//linear
var linear = (input,m,b) => {
	return input*m + b;
}
//bin
var bin = (input) => {
	if (input < 0) {
		return 0;
	} else {
		return 1;
	}
}
//tanh
var tanh = (input) => {
	return Math.tanh(input);
}
//sig
var sig  = (input) => {
	return 1 / 1 + (Math.E ** -input);
}
//leakyReLU
var leakyReLU = (input) => {
	if (input >= 0) {
		return input;
	} else {
		return input * 0.01;
	}
}
//para
var paraReLU = (input,a) => {
	if (input >= 0) {
		return input;
	} else {
		return input * a;
	}
}
//ELU
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
    gl1 = document.createElement('canvas').getContext('webgl2');
    if (gl1) {
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
///
function weiGen(lt) {
    let wei = [];
    for (let i = 0;i < lt;i++) {
        wei.push(Math.random());
    }
    return wei;
}
///
function createTexture(gl, data, width, height) {
	const tex = gl.createTexture();
	gl.bindTexture(gl.TEXTURE_2D, tex);
	gl.texImage2D(
		gl.TEXTURE_2D,
		0,
		gl.RGBA,
		width,
		height,
		0,
		gl.RGBA,
		gl.UNSIGNED_BYTE,
		new Uint8Array(data.buffer),
		null
	);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	return tex;
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
	this.bul;
	this.canvas = [];
	this.loop = 0;
	this.input;
	this.output;
	////
	if (LIB.gpuMode) {
		this.wGS = checkWebGL();
		if (this.wGS[0]) {
			this.s = this.wGS[1];
		}
	}
	////
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
			uniform sampler2D weight;
			uniform sampler2D net_input;
			uniform bool lE;
			uniform vec2 srcD;
			//float to rgba
			float shiftRight (float v, float amt) {
				v = floor(v) + 0.5;
				return floor(v / exp2(amt));
			}
			float shiftLeft (float v, float amt) {
				return floor(v * exp2(amt) + 0.5);
			}
			float maskLast (float v, float bits) {
				return mod(v, shiftLeft(1.0, bits));
			}
			float extractBits (float num, float from, float to) {
				from = floor(from + 0.5); to = floor(to + 0.5);
				return maskLast(shiftRight(num, from), to - from);
			}
			vec4 floatToRgba(float texelFloat, bool littleEndian) {
				if (texelFloat == 0.0) return vec4(0, 0, 0, 0);
				float sign = texelFloat > 0.0 ? 0.0 : 1.0;
				texelFloat = abs(texelFloat);
				float exponent = floor(log2(texelFloat));
				float biased_exponent = exponent + 127.0;
				float fraction = ((texelFloat / exp2(exponent)) - 1.0) * 8388608.0;
				float t = biased_exponent / 2.0;
				float last_bit_of_biased_exponent = fract(t) * 2.0;
				float remaining_bits_of_biased_exponent = floor(t);
				float byte4 = extractBits(fraction, 0.0, 8.0) / 255.0;
				float byte3 = extractBits(fraction, 8.0, 16.0) / 255.0;
				float byte2 = (last_bit_of_biased_exponent * 128.0 + extractBits(fraction, 16.0, 23.0)) / 255.0;
				float byte1 = (sign * 128.0 + remaining_bits_of_biased_exponent) / 255.0;
				return (
					littleEndian
					? vec4(byte4, byte3, byte2, byte1)
					: vec4(byte1, byte2, byte3, byte4)
				);
			}
			//rgba to float
			ivec4 floatsToBytes(vec4 inputFloats, bool littleEndian) {
				ivec4 bytes = ivec4(inputFloats * 255.0);
				return (
					littleEndian
					? bytes.abgr
					: bytes
				);
			}
			void bytesToBits(const in ivec4 bytes, out bool bits[32]) {
				for (int channelIndex = 0; channelIndex < 4; ++channelIndex) {
				  	float acc = float(bytes[channelIndex]);
				  	for (int indexInByte = 7; indexInByte >= 0; --indexInByte) {
						float powerOfTwo = exp2(float(indexInByte));
						bool bit = acc >= powerOfTwo;
						bits[channelIndex * 8 + (7 - indexInByte)] = bit;
						acc = mod(acc, powerOfTwo);
				  	}
				}
			}
			float getExponent(bool bits[32]) {
				const int startIndex = 1;
				const int bitStringLength = 8;
				const int endBeforeIndex = startIndex + bitStringLength;
				float acc = 0.0;
				int pow2 = bitStringLength - 1;
				for (int bitIndex = startIndex; bitIndex < endBeforeIndex; ++bitIndex) {
				  	acc += float(bits[bitIndex]) * exp2(float(pow2--));
				}
				return acc;
			}
			float getMantissa(bool bits[32], bool subnormal) {
				const int startIndex = 9;
				const int bitStringLength = 23;
				const int endBeforeIndex = startIndex + bitStringLength;
				// Leading/implicit/hidden bit convention:
				// If the number is not subnormal (with exponent 0), we add a leading 1 digit.
				float acc = float(!subnormal) * exp2(float(bitStringLength));
				int pow2 = bitStringLength - 1;
				for (int bitIndex = startIndex; bitIndex < endBeforeIndex; ++bitIndex) {
				  	acc += float(bits[bitIndex]) * exp2(float(pow2--));
				}
				return acc;
			}
			float bitsToFloat(bool bits[32]) {
				float signBit = float(bits[0]) * -2.0 + 1.0;
				float exponent = getExponent(bits);
				bool subnormal = abs(exponent - 0.0) < 0.01;
				float mantissa = getMantissa(bits, subnormal);
				float exponentBias = 127.0;
				return signBit * mantissa * exp2(exponent - exponentBias - 23.0);
			}
			float rgbaToFloat(vec4 texelRGBA, bool littleEndian) {
				ivec4 rgbaBytes = floatsToBytes(texelRGBA, littleEndian);
				bool bits[32];
				bytesToBits(rgbaBytes, bits);
				return bitsToFloat(bits);
			}
			float node_input,
			wei;
			void main() {
				vec2 texcoord = gl_FragCoord.xy / srcD;
				node_input = rgbaToFloat(texture2D(net_input,vec2(texcoord.y,0.0)),lE);
				wei = rgbaToFloat(texture2D(weight,vec2(texcoord.x,texcoord.y)),lE);
				gl_FragColor = floatToRgba(node_input * wei,lE);
			}`;
		var fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
		gl.shaderSource(fragmentShader, frag);
		gl.compileShader(fragmentShader);
		var vertexShader = gl.createShader(gl.VERTEX_SHADER);
		gl.shaderSource(vertexShader, ver);
		gl.compileShader(vertexShader);
		var program = gl.createProgram();
		gl.attachShader(program, vertexShader);
		gl.attachShader(program, fragmentShader);
		gl.linkProgram(program);
		gl.useProgram(program);
		return [gl,program,canvas];
	}
	////
	this.PropageWebGL2 = () => {

	}
	////
    this.setup = (build) => {
		this.bul = build;
		let wei;
		if (this.s == 0) {
			for (let l = 0,bl = build.length - 1,weiL = [];l < bl;l++) {
				weiL = [];
				for (let i = 0,lc = build[l + 1];i < lc;i++) {
					wei = weiGen(build[l]);
					weiL.push(wei);
				}
				this.weight.push(weiL);
			}
		} else {
			for (let l = 0,bl = build.length - 1,weiL = [];l < bl;l++) {
				wei = weiGen(build[l] * build[l + 1]);
				this.weight.push(wei);
			}
		}
		return this.weight;
    }
	////
	this.Node = (input,weight) => {
		let output = 0;
		for (let i = 0,len = weight.length;i < len;i++) {
			output += input[i] * weight[i];
		}
		return output;
	}
	////
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
	////
	this.getContextWebGL1 = (act) => {
		this.propageGPU = this.PropageWebGL1();
		this.actGPU = window[act + "WebGL1"]();
	}
	////
	this.getContextWebGL2 = (act) => {
		this.propageGPU = this.PropageWebGL2();
		this.actGPU = window[act + "WebGL2"]();
	}
	////
    this.propagation = (input) => {
		var webGL;
		if (this.s == 0) {
			let netIn = input;
			for (let l = 0,len = this.weight.length;l < len;l++) {
				netIn = this.Perceptron(netIn,this.weight[l],act,0);
			}
			return  netIn;
		} else if (this.s == 3) {//CTDSKX
			this.getContextWebGL1(act);
		} else if (this.s == 2) {
			this.getContextWebGL2(act);
		}
		if (this.s ==3) {//CTDSKX
			return () => {
				let glPro = this.propageGPU[0],
				proGramPro = this.propageGPU[1],
				canPro = this.propageGPU[2],
				bul = this.bul,
				bulLoop1,
				bulLoop,
				output;
				this.loop++;
				bulLoop1 = bul[this.loop - 1];
				bulLoop = bul[this.loop];
				const positionLoc = glPro.getAttribLocation(proGramPro, 'position');
				const wei = glPro.getUniformLocation(proGramPro, 'weight');
				const net_input = glPro.getUniformLocation(proGramPro, 'net_input');
				const srcDimensionsLoc = glPro.getUniformLocation(proGramPro, 'srcD');
				const littleE = glPro.getUniformLocation(proGramPro, 'lE');
				const buffer = glPro.createBuffer();
				glPro.bindBuffer(glPro.ARRAY_BUFFER, buffer);
				glPro.bufferData(glPro.ARRAY_BUFFER, new Float32Array([
				-1, -1,
				1, -1,
				-1,  1,
				-1,  1,
				1, -1,
				1,  1,
				]), glPro.STATIC_DRAW);
				glPro.enableVertexAttribArray(positionLoc);
				glPro.vertexAttribPointer(
					positionLoc,
					2,
					glPro.FLOAT,
					false,
					0,
					0,
				);
				glPro.uniform2fv(srcDimensionsLoc, [bulLoop1,bulLoop]);
				glPro.uniform1i(littleE, lE);
				glPro.uniform1i(wei, 0);
				glPro.activeTexture(glPro.TEXTURE0);
				glPro.bindTexture(glPro.TEXTURE_2D,createTexture(glPro,new Float32Array(this.weight[this.loop - 1]),bulLoop1,bulLoop));
				glPro.uniform1i(net_input, 1);
				glPro.activeTexture(glPro.TEXTURE1);
				glPro.bindTexture(glPro.TEXTURE_2D,createTexture(glPro,new Float32Array(this.input),bulLoop1,1));
				canPro.width = bulLoop1;
				canPro.width = bulLoop;
				glPro.drawArrays(glPro.TRIANGLES, 0, 6);
				output = new Uint8Array((bulLoop1 * bulLoop) * 4);
				glPro.readPixels(0, 0, bulLoop1,bulLoop, glPro.RGBA, glPro.UNSIGNED_BYTE, output);
				output = new Float32Array(output.buffer);
				console.log(output);
				this.input = output;
				this.output = output;
			};
		} else {
			return (input) => {
			};
		}
    }
	////
	this.run = (input,pro) => {
		this.input = input;
		loopCount = this.weight.length;
		this.loop = 0;
		pro();
		if (this.loop < loopCount) {
			requestAnimationFrame(this.run);
		}
		return this.output;
	}
	////
	this.backpropagation = (out,pOut) => {

	}
}
