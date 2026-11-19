.text
.global _start
_start:
  MOVT A0.0,#0xb000          ! coherent mailbox
  MOV D0.0,#0
  SETD [A0.0+#4],D0.0        ! last submitted sequence
wait:
  GETD D0.0,[A0.0]
  GETD D1.0,[A0.0+#4]
  CMP D0.0,D1.0
  BEQ wait
  MOV D1.1,D0.0
  .macro gpu_addr reg
    MOVT A0.1,#0x0280        ! GPU register broadcast window
    ADD A0.1,A0.1,#\reg
  .endm
  .macro reg64 reg, off
    GETL D0.0,D1.0,[A0.0+#\off]
    gpu_addr \reg
    SETL [A0.1],D0.0,D1.0
  .endm
  .macro reg32 reg, value
    MOV D0.0,#\value
    gpu_addr \reg
    SETD [A0.1],D0.0
  .endm
  reg32 0x4000,1             ! select compute scheduler
  reg32 0xf920,0             ! CDM uses MMU context zero
  reg64 0x4f8,8              ! per-core queue controls
  reg64 0x4e8,16             ! stream ring
  reg64 0x4f0,24             ! ring size
  reg64 0x598,32             ! context state
  reg32 0x5b8,0
  reg32 0x4d0,1
  reg64 0x498,40             ! compute state
  reg64 0x68,48              ! participating cores
  reg32 0x24c0,1
  reg32 0x478,1              ! start the compute stream
  SETD [A0.0+#4],D1.1
  B wait
