import struct
from collections import Counter, defaultdict
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer import Compiler, Renderer
from tinygrad.uop.ops import Ops, UOp

def _field(value:int, bits:tuple[int, ...]) -> int:
  if not 0 <= value < 1<<len(bits): raise NotImplementedError(f"S30 USC operand {value} exceeds {len(bits)} bits")
  return sum(((value >> i) & 1) << bit for i,bit in enumerate(bits))

# Sources are instruction indices. A complemented index selects a shared register directly on these arithmetic operands.
shared_bits:dict[str,tuple[int|None,...]] = {"iadd":(28,20), "imul":(28,None), "imad":(28,None,20), "fmul":(15,43), "fma":(15,43,35)}
half_bits = {"fmul":(9,10), "fadd":(9,26), "fma":(9,10,26)}
Instruction = tuple[str, tuple[int, ...], int]

def assemble(ins:list[Instruction]) -> tuple[bytes, int]:
  # An operand defined before a loop and read in its body must survive the back edge.
  last:dict[int,int] = {}
  loops = [(src[0], i) for i,(op,src,_) in enumerate(ins) if op == "back"]
  for i,(_,src,_) in enumerate(ins):
    for v in src:
      if v >= 0: last[v] = max(last.get(v, i), i, *(end for start,end in loops if v <= start <= i <= end))
  # Overlap two independent reads. Keep the first address live until the second DMA waits for both data fences.
  first:set[int] = set()
  second:set[int] = set()
  pending:int|None = None
  for i,(op,src,_) in enumerate(ins):
    if pending is not None and (pending in src or op not in ("ld", "ld8", "ld16", "imm", "imul", "imad", "iadd", "lsr", "lsl", "asr",
                                                             "and", "or", "xor", "group", "lane")): pending = None
    if op in ("ld", "ld8", "ld16"):
      if pending is not None and last.get(pending, pending) > i:
        first.add(pending)
        second.add(i)
        for v in ins[pending][1]: last[v] = max(last[v], i)
        pending = None
      else: pending = i
  expires:dict[int,list[int]] = defaultdict(list)
  for v,i in last.items(): expires[i].append(v)
  regs:dict[int,int] = {}
  labels:dict[int,int] = {}
  free, peak = set(range(64)), 0
  code = bytearray()
  fence:int|None = None
  for i,(op,src,imm) in enumerate(ins):
    if op in ("imm", "half_to_float") and i not in last: continue
    operands = tuple(regs[v] if v >= 0 else ~v for v in src)
    for v in expires[i]: free.add(regs.pop(v))
    if op not in ("st", "st8", "st16", "assign", "back", "cndst", "cndend"):
      if not free: raise NotImplementedError("S30 USC needs more than 64 live temporaries")
      dst = min(free)
      peak = max(peak, dst+1)
      if i in last:
        free.remove(dst)
        regs[i] = dst
    if op == "slot": continue
    if op == "back":
      delta = (labels[src[0]]-len(code))//4
      disp = _field(delta & 0x3fffffff,(16,14,13,15,17,18,28,39,41,44,43,40,38,37,36,35,30,29,26,19,21,24,23,22,20,27,25,32,34,33))
      code.extend((0x8000800000000c1c | disp).to_bytes(8, 'little'))  # wait wfc0 and wfc1, branch if any P0
      fence = None
      continue

    # The USC scatters register-number bits across instruction words. Each base word selects scalar 32-bit operands.
    if op == "group": word, wait, signal = 0xc00b804a | _field(imm,(18,22)) | _field(dst,(9,10,23,24,47,56)), (20,29), 11  # mov rD, lmN
    elif op in ("cndst", "cndend"): word, wait, signal = 0x8001105e if op == "cndst" else 0x8000109e, (9,11), 8
    elif op == "lane": word, wait, signal = 0x804008000482880f | imm<<14 | _field(dst,(4,16,24,33,42,45)), (55,56), 41  # mov rD, arN
    elif op in ("mov", "range", "assign"):
      dest, a = operands if op == "assign" else (dst, operands[0])
      word, wait, signal = 0xc00b80ca | _field(dest,(9,10,23,24,47,56)) | _field(a,(18,22,21,25,39,50)), (20,29), 11
    elif op in ("imul", "imad"):
      a,b = operands[:2]
      word = 0x840c0a990110200110106c20 if op == "imul" else 0x840c0a990110200910904c00 | _field(operands[2],(15,38,37,40,47,54))
      word |= _field(dst,(19,41,53,59,70,46)) | _field(a,(6,16,17,36,48,27)) | _field(b,(42,61,57,74,69,22))
      wait, signal = (29,39), 18
    elif op in ("lsr", "asr", "lsl", "and", "or", "xor"):
      a,b = operands
      word = {"lsr":0xa5b6200149801078, "asr":0x9a20824025b6200149801078, "lsl":0x9a20824025b6000149801078,
              "and":0xa136000041801078,
              "or":0x9b00c24025b6000159c01078, "xor":0x9b00c04024b6000118c01078}[op] | _field(dst,(8,14,19,33,41,42))
      word |= _field(a,(25,36,44,47,65,74)) | _field(b,(13,17,34,29,35,39))
      wait, signal = (11,18), 10
    elif op in ("sqrt", "recip", "exp2", "log2"):
      word = {"sqrt":0x80000a000200001a, "recip":0x80000a000000001a, "exp2":0x80000a010002001a, "log2":0x80000a010000001a}[op]
      word |= _field(dst,(7,8,16,20,30,34)) | _field(operands[0],(9,13,19,22,33,36))
      wait, signal = (10,15), 14
    elif op in ("half_to_float", "float_to_half"):
      a, = operands
      word = (0x804008080082880b | _field(a,(14,21,25,27,37,39)) if op == "half_to_float" else 0x8040080804c08687 | _field(a,(6,12,20,19,29,28)))
      word |= _field(dst,(4,16,24,33,42,45))
      wait, signal = (55,56), 41
    elif op == "select":
      _, t, f = operands
      word = 0xc04008080480a68b | _field(dst,(4,16,24,33,42,45)) | _field(t,(14,21,25,27,37,39)) | _field(f,(6,12,20,19,29,28))
      wait, signal = (55,56), 41
    elif op in ("u8tof", "itof", "utof", "ftoi", "ftou"):
      word = {"u8tof":0x80211818400388ca, "itof":0x80010818400388ca, "utof":0x80031808400388ca,
              "ftoi":0x80221818400388ca, "ftou":0x8040003800001818400388ca}[op]
      word |= _field(dst,(9,10,23,24,47,56)) | _field(operands[0],(18,22,21,25,39,50))
      wait, signal = (20,29), None  # conversion waits for its own dfc0
    elif op in ("iadd", "imm", "ilt", "imax", "ieq"):
      a,b = (0,imm) if op == "imm" else operands
      word = {"iadd":0x8190200010904c00, "imm":0x8190200010804c80, "ilt":0x8596240816f07d00,
              "imax":0x9592200a10b06d00, "ieq":0x8196240816f07d00}[op] | _field(dst,(19,41,53,59,70,46))
      src_bits = (15,38,37,40,47,54)
      word |= _field(a,(6,16,17,36,48,27)) | _field(b,src_bits+(66,76) if op == "imm" else src_bits)
      wait, signal = (29,39), 18
    elif op == "fma":
      a,b,c = operands
      word = 0x8040080804808609 | _field(dst,(4,16,24,33,42,45)) | _field(a,(6,12,20,19,29,28))
      word |= _field(b,(22,32,38,40,50,58)) | _field(c,(14,21,25,27,37,39))
      wait, signal = (55,56), 41
    elif op in ("fadd", "fmul", "fmax", "feq", "fgt"):
      a,b = operands
      word = {"fadd":0x804008080480a60b, "fmul":0x804008080480868d,
              "fmax":0x804318080482a60b, "feq":0xa04108080482a60b, "fgt":0xa04208080482a60b}[op]
      word |= _field(dst,(4,16,24,33,42,45)) | _field(a,(6,12,20,19,29,28))
      word |= _field(b, (14,21,25,27,37,39) if op != "fmul" else (22,32,38,40,50,58))
      wait, signal = (55,56), 41
    elif op in ("ld", "ld8", "ld16", "st", "st8", "st16"):
      addr = _field(imm,(45,17,22,30,36)) | _field(operands[0],(20,14,26,27,34,40))  # [shared pointer pair + byte offset in rN]
      if op.startswith("ld"): word, signal = 0xc00004000001858e | addr | _field(dst,(29,35,37,41,47,53)), None
      else: word, signal = 0xc000008000012a4e | addr | _field(operands[1],(25,23,28,32,38,44)), 15
      if i in first: word ^= 1<<42  # set dfc0 and continue
      if i in second: word ^= 1<<15 | 1<<21 | 1<<46  # set dfc1 and wait for dfc0 and dfc1
      if op.endswith("16"): word ^= 0x0004000000010000
      if op.endswith("8"): word ^= 0x0010000000010000
      wait = (18,19)
    else: raise ValueError(op)

    if op in half_bits: word ^= imm
    for v,bit in zip(src, shared_bits.get(op, ())):
      if v < 0 and bit is not None: word ^= 1<<bit

    if op in ("group", "mov", "range", "assign") and word >= 1<<32:
      word = (word & ~(1<<31)) | 0x80000800<<32  # one-word burst
    elif op in ("iadd", "imm", "ilt", "imax", "ieq") and word >= 1<<64:
      word = (word & ~(1<<63)) | 0x840c0a99<<64  # scalar 32-bit integer pipeline
    elif op in ("and", "lsr") and word >= 1<<64:
      word = (word & ~(1<<63)) | 0x9a008240<<64
    # Alternate ALU write fences. Single loads wait on their own data fence.
    if fence is not None: word |= 1<<wait[fence]
    fence = None if signal is None else int(fence == 0)
    if fence == 1 and signal is not None: word |= 1<<signal
    code.extend(word.to_bytes(4*((word.bit_length()+31)//32), 'little'))
    if op == "range": labels[i] = len(code)
  code.extend(struct.pack('<I', 0x8000031c | (0 if fence is None else 0x400<<fence)))  # wait and end
  return bytes(code), peak

class USCCompiler(Compiler):
  def compile(self, src:str) -> bytes: return bytes.fromhex(src)

class MTUSCRenderer(Renderer):
  compiler = USCCompiler()
  supports_float4 = has_shared = False
  shared_max = 0
  global_max, local_max = (0x7fffffff,65535,65535), (1024,1024,64)
  code_for_op = dict.fromkeys((Ops.ADD, Ops.SHL, Ops.SHR, Ops.MUL, Ops.SQRT, Ops.RECIP, Ops.EXP2, Ops.LOG2, Ops.MAX), lambda: None)
  def __init__(self, target:Target):
    if target.arch != "mp_10": raise RuntimeError("USC requires the S30")
    super().__init__(target)
  def supported_dtypes(self):
    return {dtypes.float, dtypes.half, dtypes.int, dtypes.uint, dtypes.short, dtypes.ushort, dtypes.char, dtypes.uchar, dtypes.bool}
  def render(self, uops:list[UOp]) -> str:
    uses = Counter(x for u in uops for x in u.src)
    fused:dict[UOp,tuple[UOp,UOp]] = {}
    muls:set[UOp] = set()
    for u in uops:
      if u.op in (Ops.RANGE, Ops.END): muls.clear()
      if u.op is Ops.MUL and uses[u] == 1 and u.dtype in (dtypes.float, dtypes.int, dtypes.uint): muls.add(u)
      if u.op is Ops.ADD:
        if u.src[0] in muls: fused[u] = (u.src[0],u.src[1])
        elif u.src[1] in muls: fused[u] = (u.src[1],u.src[0])
    products = {a for a,_ in fused.values()}
    has_loop = any(u.op is Ops.RANGE for u in uops)
    supported = self.supported_dtypes()
    params = sorted((u for u in uops if u.op is Ops.PARAM), key=lambda u:u.arg.slot)
    if any((p.addrspace != AddrSpace.GLOBAL or p.dtype not in supported) and
           (p.addrspace != AddrSpace.ALU or p.dtype != dtypes.int) for p in params):
      raise NotImplementedError("S30 USC requires supported global buffer arguments")
    ptrs = {p:2*i for i,p in enumerate(params)}
    ins:list[Instruction] = []
    constants:list[int] = []
    regs:dict[UOp,int] = {}
    offsets:dict[tuple[UOp,int],int] = {}
    def emit(op:str, *src:int, imm:int=0) -> int:
      for i,half_bit in enumerate(half_bits.get(op, ())):
        if ins[src[i]][0] == "half_to_float":
          src = (*src[:i], ins[src[i]][1][0], *src[i+1:])
          imm |= 1<<half_bit
      if op in shared_bits:
        if op in ("imul", "imad") and ins[src[1]][0] == "imm": src = (src[1],src[0],*src[2:])
        for i,(v,bit) in enumerate(zip(src, shared_bits[op])):
          if bit is not None and ins[v][0] == "imm" and ins[v][2] < 64:
            src = (*src[:i], ~ins[v][2], *src[i+1:])
            break
      ins.append((op, src, imm))
      return len(ins)-1
    def literal(bits:int) -> int:
      if bits not in constants: constants.append(bits)
      return emit("imm", imm=2*len(params)+constants.index(bits))
    def select(cond:int, t:int, f:int) -> int:
      return emit("select", emit("ilt", literal(0), cond), t, f)
    slots:dict[tuple[UOp,int],int] = {}
    def scale(src:int, n:int) -> int:
      if n <= 0: raise NotImplementedError("S30 USC needs a positive integer scale")
      acc = src
      for bit in bin(n)[3:]:
        acc = emit("iadd", acc, acc)
        if bit == '1': acc = emit("iadd", acc, src)
      return acc
    quot_rem:dict[tuple[UOp,UOp],tuple[int,int]] = {}
    def divide(n:UOp, d:UOp) -> tuple[int,int]:
      if (n,d) in quot_rem: return quot_rem[n,d]
      a,b = reg(n), reg(d)
      from tinygrad.codegen.decomp.op import magicgu
      if d.vmin == d.vmax and (m_shift:=magicgu(int(n.vmax), int(d.vmin)))[0]*n.vmax < 1<<32:
        m,shift = m_shift
        q = emit("lsr", emit("imul", a, literal(m)), literal(shift))
        rem = emit("iadd", a, emit("imul", q, literal(-int(d.vmin) & 0xffffffff)))
      else:
        # Shift the remainder right for the comparison so the trial divisor cannot overflow.
        q, rem = literal(0), a
        for i in reversed(range((int(n.vmax)//int(d.vmin)).bit_length())):
          trial = emit("iadd", rem, emit("imul", b, literal((-1<<i) & 0xffffffff)))
          with_bit = emit("iadd", q, literal(1<<i))
          keep = emit("ilt", emit("lsr", rem, literal(i)), b)
          rem = emit("select", keep, rem, trial)
          q = emit("select", keep, q, with_bit)
      quot_rem[n,d] = q,rem
      return q,rem
    def narrow(r:int, dtype:DType) -> int:
      if dtype.itemsize < 4 and dtypes.is_int(dtype):
        if dtypes.is_unsigned(dtype): return emit("and", r, literal((1<<(dtype.itemsize*8))-1))
        shift = 32-dtype.itemsize*8
        return emit("asr", emit("imul", r, literal(1<<shift)), literal(shift))
      return r
    def reg(u:UOp) -> int:
      if u in regs: return regs[u]
      if u._shape != () or u.dtype not in supported:
        raise NotImplementedError(f"S30 USC scalar {u.op} {u.dtype}")
      if u.op is Ops.SPECIAL and u.arg in ("gidx0", "gidx1", "gidx2"): r = emit("group", imm=int(u.arg[-1]))
      elif u.op is Ops.SPECIAL and u.arg == "lidx0": r = emit("lane")
      elif u.op is Ops.SPECIAL and u.arg in ("lidx1", "lidx2"):
        yz = emit("lane", imm=1)
        r = emit("and", yz, literal(0xffff)) if u.arg == "lidx1" else emit("lsr", yz, literal(16))
      elif u.op is Ops.AFTER: r = reg(u.src[0])
      elif u.op is Ops.PARAM and u.addrspace is AddrSpace.ALU: r = emit("imm", imm=ptrs[u])
      elif u.op is Ops.CONST or (u.op is Ops.CAST and u.src[0].op is Ops.CONST):
        val = u.val if u.op is Ops.CONST else u.src[0].val
        bits = int.from_bytes(struct.pack('<'+u.dtype.fmt, val), 'little') if u.dtype in (dtypes.half, dtypes.float) else int(val) & 0xffffffff
        return literal(bits)
      elif u.op is Ops.BITCAST and u.src[0].dtype.itemsize == u.dtype.itemsize: r = narrow(reg(u.src[0]), u.dtype)
      elif u.op is Ops.CAST:
        src = u.src[0]
        a = reg(src)
        if dtypes.is_float(src.dtype) and src.dtype != u.dtype:
          if src.dtype == dtypes.half: a = emit("half_to_float", a)
          if dtypes.is_int(u.dtype): a = emit("ftou" if u.dtype == dtypes.uint else "ftoi", a)
        elif dtypes.is_int(src.dtype) and dtypes.is_float(u.dtype):
          a = emit("u8tof" if src.dtype == dtypes.uchar else "utof" if src.dtype == dtypes.uint else "itof", a)
        elif src.dtype == dtypes.bool and dtypes.is_float(u.dtype): a = emit("itof", a)
        if u.dtype == dtypes.bool:
          zero = literal(0)
          r = emit("ieq", emit("feq" if dtypes.is_float(src.dtype) else "ieq", a, zero), zero)
        elif u.dtype == dtypes.half and src.dtype != dtypes.half: r = emit("float_to_half", a)
        else: r = narrow(a, u.dtype)
      elif u.op in (Ops.CMPEQ, Ops.CMPLT):
        a,b = (reg(x) for x in u.src)
        dt = u.src[0].dtype
        if dtypes.is_float(dt):
          if dt == dtypes.half: a,b = emit("half_to_float", a), emit("half_to_float", b)
          r = emit("ilt", literal(0), emit("feq", a, b) if u.op is Ops.CMPEQ else emit("fgt", b, a))
        else:
          if dt == dtypes.uint and u.op is Ops.CMPLT: a,b = emit("iadd", a, literal(1<<31)), emit("iadd", b, literal(1<<31))
          r = emit("ieq" if u.op is Ops.CMPEQ else "ilt", a, b)
      elif u.op in (Ops.AND, Ops.OR, Ops.XOR): r = emit(u.op.name.lower(), *(reg(x) for x in u.src))
      elif u.op is Ops.WHERE:
        cond, t, f = (reg(x) for x in u.src)
        r = select(cond, t, f)
      elif u in fused:
        mul, acc = fused[u]
        r = emit("fma" if u.dtype == dtypes.float else "imad", *(reg(x) for x in (*mul.src,acc)))
      elif u.op in (Ops.ADD, Ops.MUL, Ops.MAX) and u.dtype != dtypes.bool:
        a,b = (reg(x) for x in u.src)
        if u.dtype == dtypes.half: a,b = emit("half_to_float", a), emit("half_to_float", b)
        if u.op is Ops.MAX and u.dtype == dtypes.uint:
          test = emit("ilt", emit("iadd", a, literal(1<<31)), emit("iadd", b, literal(1<<31)))
          r = emit("select", test, b, a)
        else: r = narrow(emit(("f" if dtypes.is_float(u.dtype) else "i")+u.op.name.lower(), a, b), u.dtype)
        if u.dtype == dtypes.half: r = emit("float_to_half", r)
      elif u.op in (Ops.RECIP, Ops.SQRT, Ops.LOG2, Ops.EXP2) and u.dtype in (dtypes.half, dtypes.float):
        a = reg(u.src[0])
        if u.dtype == dtypes.half: a = emit("half_to_float", a)
        r = emit(u.op.name.lower(), a)
        if u.dtype == dtypes.half: r = emit("float_to_half", r)
      elif u.op is Ops.SHL: r = narrow(emit("lsl", *(reg(x) for x in u.src)), u.dtype)
      elif u.op is Ops.SHR: r = emit("lsr" if u.src[0].vmin >= 0 else "asr", *(reg(x) for x in u.src))
      elif u.op in (Ops.TRUNCDIV, Ops.TRUNCMOD) and u.dtype == dtypes.int and u.src[0].vmin >= 0 and u.src[1].vmin > 0:
        r = divide(*u.src)[u.op is Ops.TRUNCMOD]
      else: raise NotImplementedError(f"S30 USC cannot lower {u.op} {u.dtype}")
      regs[u] = r
      return r

    gate:UOp|None = None
    for u in uops:
      if u.op is Ops.IF: gate = u.src[0]
      elif u.op is Ops.ENDIF: gate = None
      elif u.op is Ops.RANGE:
        if u.src[0].vmin < 1 or u.src[0].vmax >= 1<<31: raise NotImplementedError("S30 USC requires a positive loop bound")
        regs[u] = emit("range", literal(0))
      elif u.op is Ops.END:
        idx = u.src[1]
        counter = regs[idx]
        advance = emit("iadd", counter, literal(1))
        emit("assign", counter, advance)
        pred = emit("ilt", counter, reg(idx.src[0]))
        emit("back", counter, pred)
      elif u.op in (Ops.LOAD, Ops.STORE):
        idx = u.src[0]
        if (len(u.src) not in ((1,3) if u.op is Ops.LOAD else (2,)) or idx.op is not Ops.INDEX or len(idx.src) != 2 or
            idx._shape != () or idx.dtype != idx.src[0].dtype):
          raise NotImplementedError("S30 USC requires a scalar buffer access")
        index = idx.src[1]
        buf = idx.src[0].without_after
        if buf.op is Ops.BUFFER and buf.addrspace == AddrSpace.REG and index.vmin == index.vmax:
          key = (buf, int(index.vmin))
          if key not in slots: slots[key] = emit("slot")
          if len(u.src) == 3: raise NotImplementedError("S30 USC masked register load")
          if u.op is Ops.STORE:
            value = reg(u.src[1])
            emit("assign", slots[key], value if gate is None else select(reg(gate), value, slots[key]))
          else: regs[u] = emit("mov", slots[key])
          continue
        if buf not in ptrs: raise NotImplementedError("S30 USC needs a global buffer or a fixed register index")
        offset = (index, buf.dtype.itemsize)
        if offset not in offsets: offsets[offset] = scale(reg(index), buf.dtype.itemsize)
        suffix = {1:"8", 2:"16", 4:""}[buf.dtype.itemsize]
        if u.op is Ops.LOAD:
          addr = offsets[offset]
          if len(u.src) == 3: addr = select(reg(u.src[2]), addr, literal(0))
          regs[u] = narrow(emit("ld"+suffix, addr, imm=ptrs[buf]), u.dtype)
          if len(u.src) == 3: regs[u] = select(reg(u.src[2]), regs[u], reg(u.src[1]))
        if u.op is Ops.LOAD and u.dtype == dtypes.bool: regs[u] = emit("and", regs[u], literal(1))
        if u.op is Ops.STORE:
          value = reg(u.src[1])
          # IF/ENDIF encloses a single store. Compute its operands before disabling lanes.
          if gate is not None: emit("cndst", emit("ilt", literal(0), reg(gate)))
          emit("st"+suffix, offsets[offset], value, imm=ptrs[buf])
          if gate is not None: emit("cndend")
      # Respect the scheduler's placement of invariant address arithmetic before nested reductions.
      elif has_loop and u not in products and u.op in (Ops.SPECIAL, Ops.TRUNCDIV, Ops.TRUNCMOD, *self.code_for_op): reg(u)
      elif u.op not in (Ops.PARAM, Ops.BUFFER, Ops.CONST, Ops.CAST, Ops.INDEX, Ops.SPECIAL, Ops.CMPEQ, Ops.CMPLT, Ops.WHERE,
                        Ops.AND, Ops.OR, Ops.XOR, Ops.BITCAST, Ops.TRUNCDIV, Ops.TRUNCMOD, *self.code_for_op, Ops.STACK, Ops.AFTER, Ops.SINK):
        raise NotImplementedError(f"S30 USC cannot lower {u.op}")
    code, temps = assemble(ins)
    # Shared registers hold pointer pairs followed by the shader's literal words.
    return (struct.pack(f'<II{len(constants)}I', temps, len(constants), *constants) + code).hex()
