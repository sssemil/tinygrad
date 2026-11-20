.text
.global _start
_start:
  MOVT A0.0,#0xb000          ! coherent mailbox
  MOVT A0.1,#0x0200          ! primary CDM
  MOVT A1.0,#0x0210          ! secondary CDM
  MOVT A0.3,#0xb004
  ADD A0.3,A0.3,#0x7000     ! lower bridge SRAM
  MOVT A1.3,#0xb009
  SUB A1.3,A1.3,#0x7800     ! ASM24 system scratch at XDATA 0xa800
  MOV D0.1,#0
wait:
  GETB D1.0,[A1.3]
  CMP D0.1,D1.0
  BNE submit
  MOV D0.0,#1024             ! leave the PCIe link idle between scratch reads
idle:
  SUBS D0.0,D0.0,#1
  BNE idle
  B wait
submit:
  MOV D0.1,D1.0
  ADD A0.2,A0.0,#16          ! timer pairs for a profiled data readback
  TST D0.1,#64
  BEQ list
  MOV A0.2,A0.3
  ADD A0.2,A0.2,#16          ! return timings alongside the bulk completion
list:
  MOVT A1.1,#0xb000
  ADD A1.1,A1.1,#0x7000     ! first descriptor, at context offset 0x8000
dispatch:
  ! Both CDMs advance their read offsets. Rearm them before using a resident stream again.
  GETD D0.0,[A1.1+#4]
  ADD A1.2,A1.1,D0.0
  MOV D0.0,#0
  SETD [A1.2+#8],D0.0
  GETD D0.0,[A1.1+#24]
  ADD A1.2,A1.2,#128
  SETD [A1.2+#8],D0.0
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
  TST D0.1,#128              ! bit 7 requests timestamps
  BEQ kick
  ! The timer mirrors bit 31 in bit 63. Retry if the two halves straddled a rollover.
start_timer:
  GETL D0.2,D1.2,[A0.1+#0x160]
  XORS D0.3,D0.2,D1.2
  BMI start_timer
kick:
  SETD [A1.0+#0x478],D0.0
  SETD [A0.1+#0x478],D0.0
complete:
  GETD D0.0,[A0.1+#0x130]
  GETD D1.0,[A1.0+#0x130]
  AND D0.0,D0.0,D1.0
  AND D0.0,D0.0,#4
  CMP D0.0,#0
  BEQ complete
  TST D0.1,#128
  BEQ next
end_timer:
  GETL D0.0,D1.0,[A0.1+#0x160]
  XORS D0.3,D0.0,D1.0
  BMI end_timer
  SETL [A0.2],D0.2,D1.2
  SETL [A0.2+#8],D0.0,D1.0
  ADD A0.2,A0.2,#16
next:
  GETD D0.0,[A1.1]
  CMP D0.0,#0
  BEQ done
  ADD A1.1,A1.1,D0.0
  B dispatch
done:
  SETD [A0.0+#4],D0.1
  TST D0.1,#64               ! bit 6 selects a token reply instead of downloaded data
  BEQ release
  SETD [A0.3],D0.1
release:
  MOVT A0.2,#0xb008
  ADD A0.2,A0.2,#0x7000     ! ASM24 completion queue
  MOVT D0.0,#1
  SETD [A0.2+#12],D0.0
  B wait
