' SMP 운영: 같은 폴더의 bat를 창 없이 실행하고 종료코드를 작업 스케줄러에 돌려준다 (인자 = bat 파일 이름)
Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
rc = CreateObject("WScript.Shell").Run("""" & dir & "\" & WScript.Arguments(0) & """", 0, True)
WScript.Quit rc
