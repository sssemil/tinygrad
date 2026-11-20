.text
.global _start
_start:
  MOVT A0.0,#0xb000          ! coherent mailbox
  MOVT A0.1,#0x0200          ! primary CDM
  MOVT A1.0,#0x0210          ! secondary CDM
wait:
  GETD D0.1,[A0.0]
  GETD D1.0,[A0.0+#4]
  CMP D0.1,D1.0
  BEQ wait
  MOV D0.0,#4
  SETD [A1.0+#0x138],D0.0
  SETD [A0.1+#0x138],D0.0
  MOV D0.0,#1
  SETD [A1.0+#0x478],D0.0
  SETD [A0.1+#0x478],D0.0
complete:
  GETD D0.0,[A0.1+#0x130]
  GETD D1.0,[A1.0+#0x130]
  AND D0.0,D0.0,D1.0
  AND D0.0,D0.0,#4
  CMP D0.0,#0
  BEQ complete
  SETD [A0.0+#4],D0.1
  B wait
