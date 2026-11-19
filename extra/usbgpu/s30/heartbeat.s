.text
.global _start
_start:
  MOVT A0.0,#0xb000
loop:
  GETD D0.0,[A0.0]
  ADD D0.0,D0.0,#1
  SETD [A0.0+#4],D0.0
  B loop
