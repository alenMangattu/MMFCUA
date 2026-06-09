import AppKit

private func clamp(_ value: CGFloat, min minValue: CGFloat, max maxValue: CGFloat) -> CGFloat {
    min(max(value, minValue), maxValue)
}

final class PromptPanel: NSPanel {
    init(rootView: NSView) {
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: 520, height: 150),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        isOpaque = false
        backgroundColor = .clear
        hasShadow = false
        level = .statusBar
        isFloatingPanel = true
        hidesOnDeactivate = false
        isMovableByWindowBackground = true
        collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]
        animationBehavior = .utilityWindow
        contentView = rootView
        applyRoundedMask()
    }

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }

    override func setFrame(_ frameRect: NSRect, display flag: Bool) {
        super.setFrame(frameRect, display: flag)
        applyRoundedMask()
    }

    private func applyRoundedMask() {
        let cornerRadius: CGFloat = 28
        contentView?.wantsLayer = true
        contentView?.layer?.cornerRadius = cornerRadius
        contentView?.layer?.cornerCurve = .continuous
        contentView?.layer?.masksToBounds = true
        contentView?.layer?.backgroundColor = NSColor.clear.cgColor

        if let frameView = contentView?.superview {
            frameView.wantsLayer = true
            frameView.layer?.backgroundColor = NSColor.clear.cgColor

            let roundedPath = CGPath(
                roundedRect: frameView.bounds,
                cornerWidth: cornerRadius,
                cornerHeight: cornerRadius,
                transform: nil
            )
            let maskLayer = CAShapeLayer()
            maskLayer.path = roundedPath
            frameView.layer?.mask = maskLayer
            frameView.layer?.shadowPath = roundedPath
            frameView.layer?.shadowColor = NSColor.black.withAlphaComponent(0.28).cgColor
            frameView.layer?.shadowOpacity = 1
            frameView.layer?.shadowRadius = 22
            frameView.layer?.shadowOffset = CGSize(width: 0, height: -8)
        }

        invalidateShadow()
    }
}

final class PromptInputView: NSVisualEffectView {
    var onSubmit: ((String) -> Void)?

    private let titleLabel = NSTextField(labelWithString: "MMFCUA")
    private let statusLabel = NSTextField(labelWithString: "Press Enter to submit  -  Esc to close")
    private let contrastTint = NSView()
    private let inputShell = NSView()
    private let inputField = NSTextField()

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        material = .sidebar
        blendingMode = .behindWindow
        state = .active
        isEmphasized = true
        appearance = NSAppearance(named: .vibrantDark)
        wantsLayer = true

        if let layer {
            layer.cornerRadius = 28
            layer.cornerCurve = .continuous
            layer.masksToBounds = true
            layer.borderWidth = 1
            layer.borderColor = NSColor.white.withAlphaComponent(0.08).cgColor
            layer.backgroundColor = NSColor.black.withAlphaComponent(0.04).cgColor
        }

        setupViews()
        setupLayout()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError() }

    func focusInput() {
        window?.makeFirstResponder(inputField)
    }

    private func setupViews() {
        titleLabel.font = .systemFont(ofSize: 20, weight: .semibold)
        titleLabel.textColor = NSColor.white.withAlphaComponent(0.96)

        statusLabel.font = .monospacedSystemFont(ofSize: 11, weight: .medium)
        statusLabel.textColor = NSColor.white.withAlphaComponent(0.60)

        contrastTint.wantsLayer = true
        contrastTint.layer?.cornerRadius = 28
        contrastTint.layer?.cornerCurve = .continuous
        contrastTint.layer?.backgroundColor = NSColor.black.withAlphaComponent(0.10).cgColor
        contrastTint.translatesAutoresizingMaskIntoConstraints = false

        inputShell.wantsLayer = true
        inputShell.layer?.cornerRadius = 16
        inputShell.layer?.cornerCurve = .continuous
        inputShell.layer?.backgroundColor = NSColor.black.withAlphaComponent(0.14).cgColor
        inputShell.layer?.borderWidth = 1
        inputShell.layer?.borderColor = NSColor.white.withAlphaComponent(0.08).cgColor
        inputShell.translatesAutoresizingMaskIntoConstraints = false

        inputField.isBordered = false
        inputField.isBezeled = false
        inputField.drawsBackground = false
        inputField.focusRingType = .none
        inputField.textColor = NSColor.white.withAlphaComponent(0.96)
        inputField.font = .systemFont(ofSize: 18, weight: .medium)
        inputField.placeholderString = "Type a primitive action..."
        inputField.translatesAutoresizingMaskIntoConstraints = false
        inputField.delegate = self
    }

    private func setupLayout() {
        let stack = NSStackView(views: [titleLabel, inputShell, statusLabel])
        stack.orientation = .vertical
        stack.spacing = 8
        stack.alignment = .leading
        stack.translatesAutoresizingMaskIntoConstraints = false

        inputShell.addSubview(inputField)
        addSubview(contrastTint)
        addSubview(stack)

        NSLayoutConstraint.activate([
            contrastTint.leadingAnchor.constraint(equalTo: leadingAnchor),
            contrastTint.trailingAnchor.constraint(equalTo: trailingAnchor),
            contrastTint.topAnchor.constraint(equalTo: topAnchor),
            contrastTint.bottomAnchor.constraint(equalTo: bottomAnchor),

            stack.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 22),
            stack.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -22),
            stack.centerYAnchor.constraint(equalTo: centerYAnchor),

            inputShell.widthAnchor.constraint(equalTo: stack.widthAnchor),
            inputShell.heightAnchor.constraint(equalToConstant: 44),

            inputField.leadingAnchor.constraint(equalTo: inputShell.leadingAnchor, constant: 14),
            inputField.trailingAnchor.constraint(equalTo: inputShell.trailingAnchor, constant: -14),
            inputField.centerYAnchor.constraint(equalTo: inputShell.centerYAnchor),
        ])
    }

    private func submit() {
        let text = inputField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        onSubmit?(text)
    }
}

extension PromptInputView: NSTextFieldDelegate {
    func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
        if commandSelector == #selector(NSResponder.insertNewline(_:)) {
            submit()
            return true
        }
        return false
    }
}

final class PromptInputAppDelegate: NSObject, NSApplicationDelegate {
    private var panel: PromptPanel?
    private var eventMonitor: Any?
    private var screenObserver: NSObjectProtocol?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let inputView = PromptInputView(frame: NSRect(x: 0, y: 0, width: 520, height: 150))
        let panel = PromptPanel(rootView: inputView)
        self.panel = panel

        inputView.onSubmit = { text in
            print(text)
            fflush(stdout)
            NSApp.terminate(nil)
        }

        placePanel(panel)
        panel.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        inputView.focusInput()

        screenObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didChangeScreenParametersNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            guard let self, let panel = self.panel else { return }
            self.placePanel(panel)
        }

        eventMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            if event.keyCode == 53 {
                NSApp.terminate(nil)
                return nil
            }
            return event
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        if let eventMonitor {
            NSEvent.removeMonitor(eventMonitor)
        }
        if let screenObserver {
            NotificationCenter.default.removeObserver(screenObserver)
        }
    }

    private func placePanel(_ panel: NSPanel) {
        guard let screen = targetScreen() else { return }
        let visible = screen.visibleFrame
        let width = clamp(visible.width * 0.30, min: 380, max: 620)
        let height = clamp(visible.height * 0.095, min: 120, max: 148)
        let bottomInset = clamp(visible.height * 0.06, min: 32, max: 72)

        panel.setFrame(
            NSRect(
                x: visible.midX - width / 2,
                y: visible.minY + bottomInset,
                width: width,
                height: height
            ),
            display: true
        )
    }

    private func targetScreen() -> NSScreen? {
        let mouse = NSEvent.mouseLocation
        return NSScreen.screens.first(where: { NSMouseInRect(mouse, $0.frame, false) })
            ?? NSScreen.main
            ?? NSScreen.screens.first
    }
}

@main
struct PromptInputApp {
    static func main() {
        let app = NSApplication.shared
        let delegate = PromptInputAppDelegate()
        app.delegate = delegate
        app.setActivationPolicy(.accessory)
        app.run()
    }
}
