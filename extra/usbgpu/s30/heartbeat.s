.text
.global _start
_start:
  MOVT A0.0,#0xb000
  ! Read the 64-bit core identification register in each S30 core's MMIO window.
  MOVT A1.0,#0x0200
  GETL D0.0,D1.0,[A1.0+#32]
  SETL [A0.0+#8],D0.0,D1.0
  MOVT A1.0,#0x0210
  GETL D0.0,D1.0,[A1.0+#32]
  SETL [A0.0+#16],D0.0,D1.0
loop:
  GETD D0.0,[A0.0]
  ADD D0.0,D0.0,#1
  SETD [A0.0+#4],D0.0
  B loop
