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
  MOVT A1.1,#0xb000
  ADD A1.1,A1.1,#0x7000     ! first descriptor, at context offset 0x8000
dispatch:
  GETL D0.0,D1.0,[A1.1+#8]
  SETL [A1.0+#0x4f8],D0.0,D1.0
  SETL [A0.1+#0x4f8],D0.0,D1.0
  GETL D0.0,D1.0,[A1.1+#16]
  SETL [A1.0+#0x4e8],D0.0,D1.0
  SETL [A0.1+#0x4e8],D0.0,D1.0
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
  GETD D0.0,[A1.1]
  CMP D0.0,#0
  BEQ done
  ADD A1.1,A1.1,D0.0
  B dispatch
done:
  SETD [A0.0+#4],D0.1
  B wait
